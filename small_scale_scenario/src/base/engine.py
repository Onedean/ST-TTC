import os
import time
import torch
import numpy as np
from tqdm import tqdm

from src.utils.metrics import masked_mape
from src.utils.metrics import masked_rmse
from src.utils.metrics import compute_all_metrics


import torch.nn as nn

class FRPlusModule(nn.Module):
    """幅相多组校正模块"""
    def __init__(self, num_nodes, freq_bins, groups=4):
        super().__init__()
        self.groups = groups
        # 按频点分组，每组大小约 freq_bins//groups
        self.group_size = freq_bins // groups
        # 幅值和相位校正参数: [groups, num_nodes, 1]
        self.lambda_amp = nn.Parameter(torch.zeros(groups, num_nodes, 1))
        self.lambda_phi = nn.Parameter(torch.zeros(groups, num_nodes, 1))

    def forward(self, y_pred):
        # y_pred: [B,1,N,T]
        B, C, N, T = y_pred.shape
        y = y_pred[:,0]  # [B,N,T]
        # FFT -> [B,N,M]
        Yf = torch.fft.rfft(y, dim=-1)
        A = torch.abs(Yf)       # 幅值
        P = torch.angle(Yf)     # 相位

        # 构造校正后的复谱
        Yf_corr = torch.zeros_like(Yf)
        for g in range(self.groups):
            start = g * self.group_size
            end = T//2+1 if g==self.groups-1 else (g+1)*self.group_size
            lam_a = self.lambda_amp[g].unsqueeze(0)  # -> [1,N,1]
            lam_p = self.lambda_phi[g].unsqueeze(0)
            # 幅相校正
            A_g = A[:,:,start:end] * (1 + lam_a)
            P_g = P[:,:,start:end] + lam_p
            # 重构该组
            Yf_corr[:,:,start:end] = A_g * torch.exp(1j * P_g)

        # iFFT 恢复时域 [B,N,T]
        y_time = torch.fft.irfft(Yf_corr, n=T, dim=-1)
        return y_time.unsqueeze(1)  # [B,1,N,T]


class BaseEngine():
    def __init__(self, device, model, dataloader, scaler, sampler, loss_fn, lrate, optimizer, \
                scheduler, clip_grad_value, max_epochs, patience, log_dir, logger, eval_method, seed):
        super().__init__()
        self._device = device
        self.model = model
        self.model.to(self._device)

        self._dataloader = dataloader
        self._scaler = scaler

        self._loss_fn = loss_fn
        self._lrate = lrate
        self._optimizer = optimizer
        self._lr_scheduler = scheduler
        self._clip_grad_value = clip_grad_value

        self._max_epochs = max_epochs
        self._patience = patience
        self._iter_cnt = 0
        self._save_path = log_dir
        self._logger = logger
        self._eval_method = eval_method
        self._seed = seed
        
        self._logger.info('The arc of model:\n{}'.format(self.model))
        
        num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in model.parameters())
        
        self._logger.info('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
        self._logger.info('The number of parameters: {}'.format(self.model.param_num())) 
        


    def _to_device(self, tensors):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)


    def _to_numpy(self, tensors):
        if isinstance(tensors, list):
            return [tensor.detach().cpu().numpy() for tensor in tensors]
        else:
            return tensors.detach().cpu().numpy()


    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [torch.tensor(array, dtype=torch.float32) for array in nparray]
        else:
            return torch.tensor(nparray, dtype=torch.float32)


    def _inverse_transform(self, tensors):
        def inv(tensor):
            return self._scaler.inverse_transform(tensor)

        if isinstance(tensors, list):
            return [inv(tensor) for tensor in tensors]
        else:
            return inv(tensors)


    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_s{}.pt'.format(self._seed)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))


    def load_model(self, save_path):
        filename = 'final_model_s{}.pt'.format(self._seed)
        self.model.load_state_dict(torch.load(os.path.join(save_path, filename), map_location=self._device))


    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader['train_loader'].shuffle()
        for X, label in tqdm(self._dataloader['train_loader'].get_iterator()):
            self._optimizer.zero_grad()

            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            
            pred = self.model(X, label)
            pred, label = self._inverse_transform([pred, label])
            
            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('Check mask value', mask_value)
            
            loss = self._loss_fn(pred, label, mask_value)
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1
            
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)


    def train(self):
        self._logger.info('Start training!')

        wait = 0
        min_loss = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse = self.train_batch()
            t2 = time.time()

            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = self.evaluate_with_norm('val')
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            message = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}'
            self._logger.info(message.format(epoch + 1, mtrain_loss, mtrain_rmse, mtrain_mape, \
                                            mvalid_loss, mvalid_rmse, mvalid_mape, \
                                            (t2 - t1), (v2 - v1), cur_lr))

            if mvalid_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info('Val loss decrease from {:.4f} to {:.4f}'.format(min_loss, mvalid_loss))
                min_loss = mvalid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch + 1, min_loss))
                    break
        self.evaluate_with_norm('test')


    def evaluate_with_ttc(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()
        
        if mode == 'val':
            preds = []
            labels = []
            with torch.no_grad():
                for X, label in tqdm(self._dataloader[mode + '_loader'].get_iterator()):
                    # X (b, t, n, f), label (b, t, n, 1)
                    X, label = self._to_device(self._to_tensor([X, label]))
                    pred = self.model(X, label)
                    pred, label = self._inverse_transform([pred, label])
                    
                    preds.append(pred.squeeze(-1).cpu())
                    labels.append(label.squeeze(-1).cpu())

            

            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if labels.min() < 1:
                mask_value = labels.min()
        else:
            T = 12
            M = T // 2 + 1
            groups = 4
            FRP = FRPlusModule(self.model.node_num, M, groups).to(self._device)
            optim = torch.optim.Adam(FRP.parameters(), lr=1e-4)
            import queue
            q = queue.Queue(maxsize=T)
            preds, labels = [], []
            t1 = time.time()
            for x, y in tqdm(self._dataloader[mode + '_loader'].get_iterator(), desc="FRPlus TTA"):
                x, y = self._to_device(self._to_tensor([x, y]))
                with torch.no_grad():
                    yb = self.model(x)
                    yb = yb.permute(0, 3, 2, 1)
                FRP.eval()
                yc = FRP(yb)
                yc = yc.permute(0, 3, 2, 1)
                pred, label = self._inverse_transform([yc, y])
                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

                q.put((x,y))
                if q.full():
                    x_o,y_o = q.get()
                    with torch.no_grad():
                        yb_o = self.model(x_o)
                        yb_o = yb_o.permute(0, 3, 2, 1)
                    FRP.train()
                    yc_o = FRP(yb_o)
                    yc_o = yc_o.permute(0, 3, 2, 1)
                    yc_o, y_o = self._inverse_transform([yc_o, y_o])
                    
                    mask_value = torch.tensor(0)
                    if y_o.min() < 1:
                        mask_value = y_o.min()
                    
                    loss = self._loss_fn(yc_o, y_o, mask_value)
                    loss.backward()
                    optim.step()
                    optim.zero_grad()
                    FRP.eval()
            t2 = time.time()

            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)
            
            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if labels.min() < 1:
                mask_value = labels.min()
        

        if mode == 'val':
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            print('Check mask value', mask_value)
            self._logger.info('-'*50 + 'TTC Test Time Results:' + '-'*50)
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))
            self._logger.info('TTC Test Time: {:.4f}s'.format(t2 - t1))


    def evaluate_with_norm(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        t1 = time.time()
        with torch.no_grad():
            for X, label in tqdm(self._dataloader[mode + '_loader'].get_iterator()):
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, label)
                pred, label = self._inverse_transform([pred, label])
                
                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())
        t2 = time.time()
        
        
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        
        # np.save('./normal_test_pred.npy', preds.unsqueeze(1).permute(0, 1, 3, 2).cpu())
        # np.save('./normal_test_true.npy', labels.unsqueeze(1).permute(0, 1, 3, 2).cpu())
        

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            print('Check mask value', mask_value)
            self._logger.info('-'*50 + 'Norm Test Time Results:' + '-'*50)
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))
            self._logger.info('Norm Test Time: {:.4f}s'.format(t2 - t1))


    def evaluate_with_tent(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()
        
        if mode == 'val':
            preds = []
            labels = []
            with torch.no_grad():
                for X, label in tqdm(self._dataloader[mode + '_loader'].get_iterator()):
                    # X (b, t, n, f), label (b, t, n, 1)
                    X, label = self._to_device(self._to_tensor([X, label]))
                    pred = self.model(X, label)
                    pred, label = self._inverse_transform([pred, label])
                    
                    preds.append(pred.squeeze(-1).cpu())
                    labels.append(label.squeeze(-1).cpu())

            

            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if labels.min() < 1:
                mask_value = labels.min()
        else:
            """
            Test-Time Entropy Minimization (TENT) adaptation
            """
            T = 12
            import queue
            q = queue.Queue(maxsize=T)
            preds, labels = [], []
            optim = torch.optim.Adam(self.model.parameters(), lr=1e-4, eps=1e-8, weight_decay=0)

            t1 = time.time()
            for x, y in tqdm(self._dataloader[mode + '_loader'].get_iterator(), desc="TENT TTA"):
                x, y = self._to_device(self._to_tensor([x, y]))

                # Prediction
                with torch.no_grad():
                    yb = self.model(x)
                # yc = self.scaler.inverse_transform(yb)
                yc = self._inverse_transform(yb)
                preds.append(yc.squeeze(-1).cpu())
                labels.append(y.squeeze(-1).cpu())

                # Enqueue for adaptation
                q.put((x, y))
                if q.full():
                    x_o, y_o = q.get()
                    # Adaptation step
                    self.model.train()
                    yb_o = self.model(x_o)
                    # yc_o = self.scaler.inverse_transform(yb_o)
                    yc_o = self._inverse_transform(yb_o)
                    mask_value = torch.tensor(0.)
                    if y_o.min() < 1:
                        mask_value = y_o.min()
                    loss = self._loss_fn(yc_o, y_o, mask_value)
                    loss.backward()
                    optim.step()
                    optim.zero_grad()
                    self.model.eval()
            t2 = time.time()

            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)
            
            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if labels.min() < 1:
                mask_value = labels.min()
        

        if mode == 'val':
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            print('Check mask value', mask_value)
            self._logger.info('-'*50 + 'TTC Test Time Results:' + '-'*50)
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))
            self._logger.info('TTC Test Time: {:.4f}s'.format(t2 - t1))



    def evaluate_with_mae(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()
        
        if mode == 'val':
            preds = []
            labels = []
            with torch.no_grad():
                for X, label in tqdm(self._dataloader[mode + '_loader'].get_iterator()):
                    # X (b, t, n, f), label (b, t, n, 1)
                    X, label = self._to_device(self._to_tensor([X, label]))
                    pred = self.model(X, label)
                    pred, label = self._inverse_transform([pred, label])
                    
                    preds.append(pred.squeeze(-1).cpu())
                    labels.append(label.squeeze(-1).cpu())

            

            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if labels.min() < 1:
                mask_value = labels.min()
        else:
            """
            Test-Time Training with Masked Autoencoders (TTT-MAE)
            """
            T = 12
            import queue
            q = queue.Queue(maxsize=T)
            preds, labels = [], []
            optim = torch.optim.Adam(self.model.parameters(), lr=1e-4, eps=1e-8, weight_decay=0)

            # Backup initial model
            import copy
            init_weights = copy.deepcopy(self.model.state_dict())

            t1 = time.time()
            for x, y in tqdm(self._dataloader[mode + '_loader'].get_iterator(), desc="TTT-MAE TTA"):
                x, y = self._to_device(self._to_tensor([x, y]))

                # Self-supervised adaptation
                self.model.train()
                feat = self.model.extract_feature(x)
                recon = self.model.feature_to_pred(feat)
                # recon = self.scaler.inverse_transform(recon)
                recon = self._inverse_transform(recon)
                mask_value = torch.tensor(0.)
                if x.min() < 1:
                    mask_value = x.min()
                loss = self._loss_fn(recon, x, mask_value)
                loss.backward()
                optim.step()
                optim.zero_grad()

                # Test with initial weights
                self.model.load_state_dict(init_weights)
                self.model.eval()
                with torch.no_grad():
                    feat_test = self.model.extract_feature(x)
                    yb = self.model.feature_to_pred(feat_test)
                # yc = self.scaler.inverse_transform(yb)
                yc = self._inverse_transform(yb)

                preds.append(yc.squeeze(-1).cpu())
                labels.append(y.squeeze(-1).cpu())

                # Restore weights for next window
                self.model.load_state_dict(init_weights)
            t2 = time.time()

            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)
            
            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if labels.min() < 1:
                mask_value = labels.min()
        

        if mode == 'val':
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            print('Check mask value', mask_value)
            self._logger.info('-'*50 + 'TTC Test Time Results:' + '-'*50)
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))
            self._logger.info('TTC Test Time: {:.4f}s'.format(t2 - t1))
