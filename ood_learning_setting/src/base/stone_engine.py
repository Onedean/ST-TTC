import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
import torch
import numpy as np
import scipy.sparse as sp
torch.autograd.set_detect_anomaly(True)
from src.utils.metrics import masked_mae, masked_rmse, masked_mape, compute_all_metrics
from tqdm import tqdm


import torch.nn as nn

class SDC_Module(nn.Module):
    """SDC Module"""
    def __init__(self, num_nodes, freq_bins, groups=4):
        super().__init__()
        self.groups = groups
        # freq_bins//groups
        self.group_size = freq_bins // groups
        # [groups, num_nodes, 1]
        self.lambda_amp = nn.Parameter(torch.zeros(groups, num_nodes, 1))
        self.lambda_phi = nn.Parameter(torch.zeros(groups, num_nodes, 1))

    def forward(self, y_pred):
        # y_pred: [B,1,N,T]
        B, C, N, T = y_pred.shape
        y = y_pred[:,0]  # [B,N,T]
        # FFT -> [B,N,M]
        Yf = torch.fft.rfft(y, dim=-1)
        A = torch.abs(Yf) 
        P = torch.angle(Yf)
        
        Yf_corr = torch.zeros_like(Yf)
        for g in range(self.groups):
            start = g * self.group_size
            end = T//2+1 if g==self.groups-1 else (g+1)*self.group_size
            lam_a = self.lambda_amp[g].unsqueeze(0)  # -> [1,N,1]
            lam_p = self.lambda_phi[g].unsqueeze(0)
            
            A_g = A[:,:,start:end] * (1 + lam_a)
            P_g = P[:,:,start:end] + lam_p
            
            Yf_corr[:,:,start:end] = A_g * torch.exp(1j * P_g)

        y_time = torch.fft.irfft(Yf_corr, n=T, dim=-1)
        return y_time.unsqueeze(1)  # [B,1,N,T]
    

class SimpleAdapter(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_nodes, 1))
        self.beta  = nn.Parameter(torch.zeros(num_nodes, 1))

    def forward(self, y_pred):
        # y_pred: [B=1, C=1, N, T]
        y = y_pred[:, 0]                  # [1, N, T]
        # 逐节点线性校正
        # y_corr = y * self.gamma + self.beta
        y_corr = torch.tanh(y) * self.gamma + self.beta
        return y_corr.unsqueeze(1)        # [1,1,N,T]



class KrigingEngine():
    def __init__(self, device, model, adj, node, sem, order, horizon, dataloader, scaler, sampler, loss_fn, lrate, optimizer, \
                 scheduler, mask_s, optimizer_ge_s, scheduler_ge_s, mask_t, optimizer_ge_t, scheduler_ge_t, 
                 clip_grad_value, max_epochs, patience, log_dir, logger, seed, alpha, beta, beta0, year):
        super().__init__()
        self._device = device
        self.model = model
        self.model.to(self._device)
        
        self._mask_s = mask_s
        self._mask_s.to(self._device)

        self._mask_t = mask_t
        self._mask_t.to(self._device)

        self._adj = adj
        self._node = node
        self._sem = sem
        self._order = order
        self._horizon = horizon
        self._dataloader = dataloader
        self._scaler = scaler

        self._loss_fn = loss_fn
        self._lrate = lrate
        self._optimizer = optimizer
        self._lr_scheduler = scheduler

        self._mask_s = mask_s
        self._optimizer_ge_s = optimizer_ge_s,
        self._lrscheduler_ge_s = scheduler_ge_s,

        self._mask_t = mask_t
        self._optimizer_ge_t = optimizer_ge_t,
        self._lrscheduler_ge_t = scheduler_ge_t,

        self._clip_grad_value = clip_grad_value

        self._max_epochs = max_epochs
        self._patience = patience
        self._iter_cnt = 0
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed
        self._alpha = alpha
        self._beta = beta
        self._beta0 = beta0
        self.year = year

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


    def _inverse_transform(self, tensors, cat):
        def inv(tensor):
            return self._scaler[cat].inverse_transform(tensor)

        if isinstance(tensors, list):
            return [inv(tensor) for tensor in tensors]
        else:
            return inv(tensors)

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_s{}_{}-{}.pt'.format(self._seed, self.year, self.year+1)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))


    def load_model(self, save_path):
        filename = 'final_model_s{}_{}-{}.pt'.format(self._seed, self.year, self.year+1)
        # self.model.load_state_dict(torch.load(
        #     os.path.join(save_path, filename)))  
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename), map_location=self._device))  


    def train_batch(self):
        self.model.train()
        train_loss1 = []
        train_mape1 = []
        train_rmse1 = []
        train_loss2 = []
        train_mape2 = []
        train_rmse2 = []
        # before or after shuffle
        # self._dataloader['train_loader'].shuffle_batch()
        self._dataloader['train_loader'].shuffle()
        for X, label in tqdm(self._dataloader['train_loader'].get_iterator()):
            self._optimizer.zero_grad()
            self._optimizer_ge_s[0].zero_grad()
            self._optimizer_ge_t[0].zero_grad()

            # S_delta: noise for sem
            spatial_noise = True
            sem = self._sem['train']
            sem = torch.stack([sem]*X.shape[0], dim=0)

            if spatial_noise:
                mean = 0 
                std = 1
                sem = torch.add(sem, self._to_device(torch.normal(mean, std, sem.shape)))
                sem = torch.clamp(sem, min=0)
            X = X[:, :, self._node['train_node'], :]
            label1 = label[..., self._node['train_observed_node'], :]
            label2 = label[..., self._node['train_unobserved_node'], :]
            
            X, label1, label2 = self._to_device(self._to_tensor([X, label1, label2]))
            pred1, pred2, log_p = self.model(X, sem, [self._mask_s, self._mask_t])
            pred1, pred2, label1, label2 = self._inverse_transform([pred1, pred2, label1, label2], 'train')
            mask_value1 = torch.tensor(0)
            mask_value2 = torch.tensor(0)
            if label1.min() < 1:
                mask_value1 = label1.min()
            if label2.min() < 1:
                mask_value2 = label2.min()
            if self._iter_cnt == 0:
                print('Check mask value ob: ', mask_value1)
                print('Check mask value un: ', mask_value2)
            mape1 = masked_mape(pred1, label1, mask_value1).item()
            rmse1 = masked_rmse(pred1, label1, mask_value1).item()
            mape2 = masked_mape(pred2, label2, mask_value2).item()
            rmse2 = masked_rmse(pred2, label2, mask_value2).item()

            loss_ob1, var_ob = self._loss_fn(pred1, label1.unsqueeze(0), mask_value1)
            loss_un, var_un = self._loss_fn(pred2, label2.unsqueeze(0), mask_value2)            
            
            loss = (loss_ob1*label1.shape[-2] + loss_un*label2.shape[-2])/(label1.shape[-2]+label2.shape[-2])
            var = var_ob + var_un
            loss = loss + var
            loss.backward()

            log_p_s = - log_p[0] * var.detach()
            log_p_s.backward()

            log_p_t = - log_p[1] * var.detach()
            log_p_t.backward()

            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()
            self._optimizer_ge_s[0].step()
            self._optimizer_ge_t[0].step()


            train_loss1.append(loss_ob1.item())
            train_mape1.append(mape1)
            train_rmse1.append(rmse1)
            train_loss2.append(loss_un.item())
            train_mape2.append(mape2)
            train_rmse2.append(rmse2)

            self._iter_cnt += 1

        return np.mean(train_loss1), np.mean(train_mape1), np.mean(train_rmse1), np.mean(train_loss2), np.mean(train_mape2), np.mean(train_rmse2)

    def train(self):
        self._logger.info('Start training!')
        wait = 0
        min_loss = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss1, mtrain_mape1, mtrain_rmse1, mtrain_loss2, mtrain_mape2, mtrain_rmse2 = self.train_batch()
            t2 = time.time()

            v1 = time.time()
            mvalid_loss1, mvalid_mape1, mvalid_rmse1, mvalid_loss2, mvalid_mape2, mvalid_rmse2, loss = self.evaluate('val')
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()
            
            message_train_ob = 'Epoch: {:03d}, Train Loss1: {:.4f}, Train RMSE_ob1: {:.4f}, Train MAPE_ob1: {:.4f}, Train Time: {:.4f}s/epoch, LR: {:.4e}'
            self._logger.info(message_train_ob.format(epoch + 1, mtrain_loss1, mtrain_rmse1, mtrain_mape1, (t2 - t1), cur_lr))

            message_train_un = 'Epoch: {:03d}, Train Loss3: {:.4f}, Train RMSE_un: {:.4f}, Train MAPE_un: {:.4f}'
            self._logger.info(message_train_un.format(epoch + 1, mtrain_loss2, mtrain_rmse2, mtrain_mape2))
            message_val_ob = 'Epoch: {:03d}, Valid Loss1: {:.4f}, Valid RMSE_ob1: {:.4f}, Valid MAPE_ob1: {:.4f}, Valid Time: {:.4f}s'
            self._logger.info(message_val_ob.format(epoch + 1, mvalid_loss1, mvalid_rmse1, mvalid_mape1, (v2-v1)))
            
            message_val_un = 'Epoch: {:03d}, Valid Loss3: {:.4f}, Valid RMSE_un: {:.4f}, Valid MAPE_un: {:.4f}'
            self._logger.info(message_val_un.format(epoch + 1, mvalid_loss2, mvalid_rmse2, mvalid_mape2))

            if loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info('Val loss decrease from {:.4f} to {:.4f}'.format(min_loss, loss))
                min_loss = loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch + 1, min_loss))
                    break
                   
        self.evaluate('test')


    def evaluate(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds1 = []
        preds2 = []
        labels1 = []
        labels2 = []
        with torch.no_grad():
            for X, label in tqdm(self._dataloader[mode + '_loader'].get_iterator()):
                X = X[:, :, self._node[mode + '_node'], :]
                sem = self._sem[mode]
                sem = torch.stack([sem]*X.shape[0], dim=0)
                label1 = label[:, :, self._node[mode + '_observed_node'], :]
                label2 = label[:, :, self._node[mode + '_unobserved_node'], :]

                X, label1, label2 = self._to_device(self._to_tensor([X, label1, label2]))
                pred1, pred2, _ = self.model(X, sem)
                pred1, pred2, label1, label2 = self._inverse_transform([pred1, pred2, label1, label2], mode)

                preds1.append(pred1.squeeze(0).squeeze(-1).cpu())
                preds2.append(pred2.squeeze(0).squeeze(-1).cpu())
                labels1.append(label1.squeeze(-1).cpu())
                labels2.append(label2.squeeze(-1).cpu())

        preds1 = torch.cat(preds1, dim=0)
        preds2 = torch.cat(preds2, dim=0)
        labels1 = torch.cat(labels1, dim=0)
        labels2 = torch.cat(labels2, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value1 = torch.tensor(0)
        mask_value2 = torch.tensor(0)
        if labels1.min() < 1:
            mask_value1 = labels1.min()
        if labels2.min() < 1:
            mask_value2 = labels2.min()
        if mode == 'val':
            mae1 = masked_mae(preds1, labels1, mask_value1).item()
            mape1 = masked_mape(preds1, labels1, mask_value1).item()
            rmse1 = masked_rmse(preds1, labels1, mask_value1).item()
            mae2 = masked_mae(preds2, labels2, mask_value2).item()
            mape2 = masked_mape(preds2, labels2, mask_value2).item()
            rmse2 = masked_rmse(preds2, labels2, mask_value2).item()
            loss = (mae1*labels1.shape[-1] + mae2*labels2.shape[-1])/(labels1.shape[-1]+labels2.shape[-1])
            return mae1, mape1, rmse1, mae2, mape2, rmse2, loss
        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            
            test_mae1 = []
            test_mape1 = []
            test_rmse1 = []

            test_mae2 = []
            test_mape2 = []
            test_rmse2 = []
            print('Check mask value ob1: ', mask_value1)
            print('Check mask value un: ', mask_value2)
            for i in range(self.model.horizon):
                res1 = compute_all_metrics(preds1[:,i,:], labels1[:,i,:], mask_value1)
                log1 = 'Horizon {:d}, Test MAE_ob: {:.4f}, Test RMSE_ob: {:.4f}, Test MAPE_ob: {:.4f}'
                self._logger.info(log1.format(i + 1, res1[0], res1[2], res1[1]))
                test_mae1.append(res1[0])
                test_mape1.append(res1[1])
                test_rmse1.append(res1[2])

                res2 = compute_all_metrics(preds2[:,i,:], labels2[:,i,:], mask_value2)
                log2 = 'Horizon {:d}, Test MAE_un: {:.4f}, Test RMSE_un: {:.4f}, Test MAPE_un: {:.4f}'
                self._logger.info(log2.format(i + 1, res2[0], res2[2], res2[1]))
                test_mae2.append(res2[0])
                test_mape2.append(res2[1])
                test_rmse2.append(res2[2])

                num_node_ob = len(self._node['test_observed_node'])
                num_node_un = len(self._node['test_unobserved_node'])
                mae = (res1[0] * num_node_ob + res2[0] * num_node_un) / (num_node_ob + num_node_un)
                rmse = (res1[2] * num_node_ob + res2[2] * num_node_un) / (num_node_ob + num_node_un)
                mape = (res1[1] * num_node_ob + res2[1] * num_node_un) / (num_node_ob + num_node_un)
                res2 = compute_all_metrics(preds2[:,i,:], labels2[:,i,:], mask_value2)
                log = 'Horizon {:d}, Test MAEall: {:.4f}, Test RMSEall: {:.4f}, Test MAPEall: {:.4f}'
                self._logger.info(log.format(i + 1, mae, rmse, mape))
                test_mae.append(mae)
                test_mape.append(mape)
                test_rmse.append(rmse)

            log = 'Average Test MAEall: {:.4f}, Test RMSEall: {:.4f}, Test MAPEall: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))

            log = 'Average Test MAE_ob1: {:.4f}, Test RMSE_ob: {:.4f}, Test MAPE_ob: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae1), np.mean(test_rmse1), np.mean(test_mape1)))

            log = 'Average Test MAE_un: {:.4f}, Test RMSE_un: {:.4f}, Test MAPE_un: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae2), np.mean(test_rmse2), np.mean(test_mape2)))



    def evaluate_with_ttc(self, mode, group, sd_lr):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds1 = []
        preds2 = []
        labels1 = []
        labels2 = []
        
        T = 12
        M = T // 2 + 1
        groups = int(group)
        node_num = len(self._node[mode + '_observed_node']) + len(self._node[mode + '_unobserved_node'])
        FRP = SDC_Module(node_num, M, groups).to(self._device)
        # FRP = SimpleAdapter(node_num).to(self._device)
        optim = torch.optim.Adam(FRP.parameters(), lr=sd_lr)
        import queue
        q = queue.Queue(maxsize=T)
        
        for x, label in tqdm(self._dataloader[mode + '_loader'].get_iterator()):
            X = x[:, :, self._node[mode + '_node'], :]                
            sem = self._sem[mode]
            sem = torch.stack([sem]*X.shape[0], dim=0)
            label1 = label[:, :, self._node[mode + '_observed_node'], :]
            label2 = label[:, :, self._node[mode + '_unobserved_node'], :]

            X, label1, label2 = self._to_device(self._to_tensor([X, label1, label2]))
            
            with torch.no_grad():
                yb1, yb2, _ = self.model(X, sem)
                yb1 = yb1.squeeze(0)
                yb2 = yb2.squeeze(0)
                yb1 = yb1.permute(0, 3, 2, 1)
                yb2 = yb2.permute(0, 3, 2, 1)
                yb = torch.cat([yb1, yb2], dim=2)
            
                FRP.eval()
                yc = FRP(yb)
                yc = yc.permute(0, 3, 2, 1)
            
            yc1, yc2 = torch.split(yc, [len(self._node[mode + '_observed_node']), len(self._node[mode + '_unobserved_node'])], dim=2)
            yc1 = yc1.unsqueeze(0)
            yc2 = yc2.unsqueeze(0)
            
            pred1, pred2, label1, label2 = self._inverse_transform([yc1, yc2, label1, label2], mode)
            
            preds1.append(pred1.squeeze(0).squeeze(-1).cpu())
            preds2.append(pred2.squeeze(0).squeeze(-1).cpu())
            labels1.append(label1.squeeze(-1).cpu())
            labels2.append(label2.squeeze(-1).cpu())

            q.put((x, label))
            
            if q.full():
                x_o, y_o = q.get()
                
                x_o = x_o[:, :, self._node[mode + '_node'], :]
                sem_o = self._sem[mode]
                sem_o = torch.stack([sem_o]*x_o.shape[0], dim=0)
                label1_o = y_o[:, :, self._node[mode + '_observed_node'], :]
                label2_o = y_o[:, :, self._node[mode + '_unobserved_node'], :]
                
                x_o, label1_o, label2_o = self._to_device(self._to_tensor([x_o, label1_o, label2_o]))
                
                FRP.train()
                optim.zero_grad()
                with torch.no_grad():
                    yb_o1, yb_o2, _ = self.model(x_o, sem_o)
                    yb_o1 = yb_o1.squeeze(0)
                    yb_o2 = yb_o2.squeeze(0)
                    yb_o1 = yb_o1.permute(0, 3, 2, 1)
                    yb_o2 = yb_o2.permute(0, 3, 2, 1)
                    yb_o = torch.cat([yb_o1, yb_o2], dim=2)
                
                yc_o = FRP(yb_o)
                yc_o = yc_o.permute(0, 3, 2, 1)
                
                y_o = torch.cat([label1_o, label2_o], dim=-2)
                
                yc_o, y_o = self._inverse_transform([yc_o, y_o], mode)
                
                mask_value = torch.tensor(0)
                if y_o.min() < 1:
                    mask_value = y_o.min()
                
                loss = masked_mae(yc_o, y_o, mask_value)
                loss.backward()
                optim.step()
            

        preds1 = torch.cat(preds1, dim=0)
        preds2 = torch.cat(preds2, dim=0)
        labels1 = torch.cat(labels1, dim=0)
        labels2 = torch.cat(labels2, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value1 = torch.tensor(0)
        mask_value2 = torch.tensor(0)
        if labels1.min() < 1:
            mask_value1 = labels1.min()
        if labels2.min() < 1:
            mask_value2 = labels2.min()
        if mode == 'val':
            mae1 = masked_mae(preds1, labels1, mask_value1).item()
            mape1 = masked_mape(preds1, labels1, mask_value1).item()
            rmse1 = masked_rmse(preds1, labels1, mask_value1).item()
            mae2 = masked_mae(preds2, labels2, mask_value2).item()
            mape2 = masked_mape(preds2, labels2, mask_value2).item()
            rmse2 = masked_rmse(preds2, labels2, mask_value2).item()
            loss = (mae1*labels1.shape[-1] + mae2*labels2.shape[-1])/(labels1.shape[-1]+labels2.shape[-1])
            return mae1, mape1, rmse1, mae2, mape2, rmse2, loss
        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            
            test_mae1 = []
            test_mape1 = []
            test_rmse1 = []

            test_mae2 = []
            test_mape2 = []
            test_rmse2 = []
            print('Check mask value ob1: ', mask_value1)
            print('Check mask value un: ', mask_value2)
            for i in range(self.model.horizon):
                res1 = compute_all_metrics(preds1[:,i,:], labels1[:,i,:], mask_value1)
                log1 = 'Horizon {:d}, Test MAE_ob: {:.4f}, Test RMSE_ob: {:.4f}, Test MAPE_ob: {:.4f}'
                self._logger.info(log1.format(i + 1, res1[0], res1[2], res1[1]))
                test_mae1.append(res1[0])
                test_mape1.append(res1[1])
                test_rmse1.append(res1[2])

                res2 = compute_all_metrics(preds2[:,i,:], labels2[:,i,:], mask_value2)
                log2 = 'Horizon {:d}, Test MAE_un: {:.4f}, Test RMSE_un: {:.4f}, Test MAPE_un: {:.4f}'
                self._logger.info(log2.format(i + 1, res2[0], res2[2], res2[1]))
                test_mae2.append(res2[0])
                test_mape2.append(res2[1])
                test_rmse2.append(res2[2])

                num_node_ob = len(self._node['test_observed_node'])
                num_node_un = len(self._node['test_unobserved_node'])
                mae = (res1[0] * num_node_ob + res2[0] * num_node_un) / (num_node_ob + num_node_un)
                rmse = (res1[2] * num_node_ob + res2[2] * num_node_un) / (num_node_ob + num_node_un)
                mape = (res1[1] * num_node_ob + res2[1] * num_node_un) / (num_node_ob + num_node_un)
                res2 = compute_all_metrics(preds2[:,i,:], labels2[:,i,:], mask_value2)
                log = 'Horizon {:d}, Test MAEall: {:.4f}, Test RMSEall: {:.4f}, Test MAPEall: {:.4f}'
                self._logger.info(log.format(i + 1, mae, rmse, mape))
                test_mae.append(mae)
                test_mape.append(mape)
                test_rmse.append(rmse)

            log = 'Average Test MAEall: {:.4f}, Test RMSEall: {:.4f}, Test MAPEall: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))

            log = 'Average Test MAE_ob1: {:.4f}, Test RMSE_ob: {:.4f}, Test MAPE_ob: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae1), np.mean(test_rmse1), np.mean(test_mape1)))

            log = 'Average Test MAE_un: {:.4f}, Test RMSE_un: {:.4f}, Test MAPE_un: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae2), np.mean(test_rmse2), np.mean(test_mape2)))



    def evaluate_with_ttc_random(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()
        
        import random
        from collections import deque

        preds1 = []
        preds2 = []
        labels1 = []
        labels2 = []

        T = 12
        M = T // 2 + 1
        groups = 4
        node_num = len(self._node[mode + '_observed_node']) + len(self._node[mode + '_unobserved_node'])
        FRP = SDC_Module(node_num, M, groups).to(self._device)
        optim = torch.optim.Adam(FRP.parameters(), lr=1e-4)
        history = deque(maxlen=8000)  # 存储历史样本

        for x, label in tqdm(self._dataloader[mode + '_loader'].get_iterator()):
            X = x[:, :, self._node[mode + '_node'], :]
            sem = self._sem[mode]
            sem = torch.stack([sem] * X.shape[0], dim=0)
            label1 = label[:, :, self._node[mode + '_observed_node'], :]
            label2 = label[:, :, self._node[mode + '_unobserved_node'], :]

            X, label1, label2 = self._to_device(self._to_tensor([X, label1, label2]))

            with torch.no_grad():
                yb1, yb2, _ = self.model(X, sem)
                yb1 = yb1.squeeze(0).permute(0, 3, 2, 1)
                yb2 = yb2.squeeze(0).permute(0, 3, 2, 1)
                yb = torch.cat([yb1, yb2], dim=2)

                FRP.eval()
                yc = FRP(yb)
                yc = yc.permute(0, 3, 2, 1)

            yc1, yc2 = torch.split(yc, [len(self._node[mode + '_observed_node']), len(self._node[mode + '_unobserved_node'])], dim=2)
            yc1 = yc1.unsqueeze(0)
            yc2 = yc2.unsqueeze(0)

            pred1, pred2, label1, label2 = self._inverse_transform([yc1, yc2, label1, label2], mode)

            preds1.append(pred1.squeeze(0).squeeze(-1).cpu())
            preds2.append(pred2.squeeze(0).squeeze(-1).cpu())
            labels1.append(label1.squeeze(-1).cpu())
            labels2.append(label2.squeeze(-1).cpu())

            history.append((x, label))  # 存储当前样本

            if len(history) >= T+1:
                
                if len(history) > 600:
                    x_o, y_o = random.choice(list(history)[:500])  # 随机选择一个样本
                else:
                    x_o, y_o = random.choice(list(history)[:-T])  # 随机选择一个样本

                x_o = x_o[:, :, self._node[mode + '_node'], :]
                sem_o = self._sem[mode]
                sem_o = torch.stack([sem_o] * x_o.shape[0], dim=0)
                label1_o = y_o[:, :, self._node[mode + '_observed_node'], :]
                label2_o = y_o[:, :, self._node[mode + '_unobserved_node'], :]

                x_o, label1_o, label2_o = self._to_device(self._to_tensor([x_o, label1_o, label2_o]))

                FRP.train()
                optim.zero_grad()
                with torch.no_grad():
                    yb_o1, yb_o2, _ = self.model(x_o, sem_o)
                    yb_o1 = yb_o1.squeeze(0).permute(0, 3, 2, 1)
                    yb_o2 = yb_o2.squeeze(0).permute(0, 3, 2, 1)
                    yb_o = torch.cat([yb_o1, yb_o2], dim=2)

                yc_o = FRP(yb_o)
                yc_o = yc_o.permute(0, 3, 2, 1)

                y_o = torch.cat([label1_o, label2_o], dim=-2)

                yc_o, y_o = self._inverse_transform([yc_o, y_o], mode)

                mask_value = torch.tensor(0)
                if y_o.min() < 1:
                    mask_value = y_o.min()

                loss = masked_mae(yc_o, y_o, mask_value)
                loss.backward()
                optim.step()



        preds1 = torch.cat(preds1, dim=0)
        preds2 = torch.cat(preds2, dim=0)
        labels1 = torch.cat(labels1, dim=0)
        labels2 = torch.cat(labels2, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value1 = torch.tensor(0)
        mask_value2 = torch.tensor(0)
        if labels1.min() < 1:
            mask_value1 = labels1.min()
        if labels2.min() < 1:
            mask_value2 = labels2.min()
        if mode == 'val':
            mae1 = masked_mae(preds1, labels1, mask_value1).item()
            mape1 = masked_mape(preds1, labels1, mask_value1).item()
            rmse1 = masked_rmse(preds1, labels1, mask_value1).item()
            mae2 = masked_mae(preds2, labels2, mask_value2).item()
            mape2 = masked_mape(preds2, labels2, mask_value2).item()
            rmse2 = masked_rmse(preds2, labels2, mask_value2).item()
            loss = (mae1*labels1.shape[-1] + mae2*labels2.shape[-1])/(labels1.shape[-1]+labels2.shape[-1])
            return mae1, mape1, rmse1, mae2, mape2, rmse2, loss
        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            
            test_mae1 = []
            test_mape1 = []
            test_rmse1 = []

            test_mae2 = []
            test_mape2 = []
            test_rmse2 = []
            print('Check mask value ob1: ', mask_value1)
            print('Check mask value un: ', mask_value2)
            for i in range(self.model.horizon):
                res1 = compute_all_metrics(preds1[:,i,:], labels1[:,i,:], mask_value1)
                log1 = 'Horizon {:d}, Test MAE_ob: {:.4f}, Test RMSE_ob: {:.4f}, Test MAPE_ob: {:.4f}'
                self._logger.info(log1.format(i + 1, res1[0], res1[2], res1[1]))
                test_mae1.append(res1[0])
                test_mape1.append(res1[1])
                test_rmse1.append(res1[2])

                res2 = compute_all_metrics(preds2[:,i,:], labels2[:,i,:], mask_value2)
                log2 = 'Horizon {:d}, Test MAE_un: {:.4f}, Test RMSE_un: {:.4f}, Test MAPE_un: {:.4f}'
                self._logger.info(log2.format(i + 1, res2[0], res2[2], res2[1]))
                test_mae2.append(res2[0])
                test_mape2.append(res2[1])
                test_rmse2.append(res2[2])

                num_node_ob = len(self._node['test_observed_node'])
                num_node_un = len(self._node['test_unobserved_node'])
                mae = (res1[0] * num_node_ob + res2[0] * num_node_un) / (num_node_ob + num_node_un)
                rmse = (res1[2] * num_node_ob + res2[2] * num_node_un) / (num_node_ob + num_node_un)
                mape = (res1[1] * num_node_ob + res2[1] * num_node_un) / (num_node_ob + num_node_un)
                res2 = compute_all_metrics(preds2[:,i,:], labels2[:,i,:], mask_value2)
                log = 'Horizon {:d}, Test MAEall: {:.4f}, Test RMSEall: {:.4f}, Test MAPEall: {:.4f}'
                self._logger.info(log.format(i + 1, mae, rmse, mape))
                test_mae.append(mae)
                test_mape.append(mape)
                test_rmse.append(rmse)

            log = 'Average Test MAEall: {:.4f}, Test RMSEall: {:.4f}, Test MAPEall: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))

            log = 'Average Test MAE_ob1: {:.4f}, Test RMSE_ob: {:.4f}, Test MAPE_ob: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae1), np.mean(test_rmse1), np.mean(test_mape1)))

            log = 'Average Test MAE_un: {:.4f}, Test RMSE_un: {:.4f}, Test MAPE_un: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae2), np.mean(test_rmse2), np.mean(test_mape2)))




    def evaluate_with_ttc_retrieval(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()
        
        import numpy as np
        from collections import deque
        
        preds1 = []
        preds2 = []
        labels1 = []
        labels2 = []

        T = 12
        M = T // 2 + 1
        groups = 4
        node_num = len(self._node[mode + '_observed_node']) + len(self._node[mode + '_unobserved_node'])
        FRP = SDC_Module(node_num, M, groups).to(self._device)
        optim = torch.optim.Adam(FRP.parameters(), lr=1e-4)
        history = deque(maxlen=1000)  # 存储历史样本

        for x, label in tqdm(self._dataloader[mode + '_loader'].get_iterator()):
            X = x[:, :, self._node[mode + '_node'], :]
            sem = self._sem[mode]
            sem = torch.stack([sem] * X.shape[0], dim=0)
            label1 = label[:, :, self._node[mode + '_observed_node'], :]
            label2 = label[:, :, self._node[mode + '_unobserved_node'], :]

            X, label1, label2 = self._to_device(self._to_tensor([X, label1, label2]))

            with torch.no_grad():
                yb1, yb2, _ = self.model(X, sem)
                yb1 = yb1.squeeze(0).permute(0, 3, 2, 1)
                yb2 = yb2.squeeze(0).permute(0, 3, 2, 1)
                yb = torch.cat([yb1, yb2], dim=2)

                FRP.eval()
                yc = FRP(yb)
                yc = yc.permute(0, 3, 2, 1)

            yc1, yc2 = torch.split(yc, [len(self._node[mode + '_observed_node']), len(self._node[mode + '_unobserved_node'])], dim=2)
            yc1 = yc1.unsqueeze(0)
            yc2 = yc2.unsqueeze(0)

            pred1, pred2, label1, label2 = self._inverse_transform([yc1, yc2, label1, label2], mode)

            preds1.append(pred1.squeeze(0).squeeze(-1).cpu())
            preds2.append(pred2.squeeze(0).squeeze(-1).cpu())
            labels1.append(label1.squeeze(-1).cpu())
            labels2.append(label2.squeeze(-1).cpu())

            history.append((x, label))  # 存储当前样本

            if len(history) >= T + 1:
                # 计算当前样本与历史样本的相似度
                current_sample = x[:, :, self._node[mode + '_node'], :].flatten()
                similarities = []
                for hist_x, _ in list(history)[:-T]:
                    hist_sample = hist_x[:, :, self._node[mode + '_node'], :].flatten()
                    similarity = np.dot(current_sample, hist_sample) / (np.linalg.norm(current_sample) * np.linalg.norm(hist_sample) + 1e-8)
                    similarities.append(similarity)
                max_index = np.argmax(similarities)
                x_o, y_o = list(history)[max_index]

                x_o = x_o[:, :, self._node[mode + '_node'], :]
                sem_o = self._sem[mode]
                sem_o = torch.stack([sem_o] * x_o.shape[0], dim=0)
                label1_o = y_o[:, :, self._node[mode + '_observed_node'], :]
                label2_o = y_o[:, :, self._node[mode + '_unobserved_node'], :]

                x_o, label1_o, label2_o = self._to_device(self._to_tensor([x_o, label1_o, label2_o]))

                FRP.train()
                optim.zero_grad()
                with torch.no_grad():
                    yb_o1, yb_o2, _ = self.model(x_o, sem_o)
                    yb_o1 = yb_o1.squeeze(0).permute(0, 3, 2, 1)
                    yb_o2 = yb_o2.squeeze(0).permute(0, 3, 2, 1)
                    yb_o = torch.cat([yb_o1, yb_o2], dim=2)

                yc_o = FRP(yb_o)
                yc_o = yc_o.permute(0, 3, 2, 1)

                y_o = torch.cat([label1_o, label2_o], dim=-2)

                yc_o, y_o = self._inverse_transform([yc_o, y_o], mode)

                mask_value = torch.tensor(0)
                if y_o.min() < 1:
                    mask_value = y_o.min()

                loss = masked_mae(yc_o, y_o, mask_value)
                loss.backward()
                optim.step()

        preds1 = torch.cat(preds1, dim=0)
        preds2 = torch.cat(preds2, dim=0)
        labels1 = torch.cat(labels1, dim=0)
        labels2 = torch.cat(labels2, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value1 = torch.tensor(0)
        mask_value2 = torch.tensor(0)
        if labels1.min() < 1:
            mask_value1 = labels1.min()
        if labels2.min() < 1:
            mask_value2 = labels2.min()
        if mode == 'val':
            mae1 = masked_mae(preds1, labels1, mask_value1).item()
            mape1 = masked_mape(preds1, labels1, mask_value1).item()
            rmse1 = masked_rmse(preds1, labels1, mask_value1).item()
            mae2 = masked_mae(preds2, labels2, mask_value2).item()
            mape2 = masked_mape(preds2, labels2, mask_value2).item()
            rmse2 = masked_rmse(preds2, labels2, mask_value2).item()
            loss = (mae1*labels1.shape[-1] + mae2*labels2.shape[-1])/(labels1.shape[-1]+labels2.shape[-1])
            return mae1, mape1, rmse1, mae2, mape2, rmse2, loss
        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            
            test_mae1 = []
            test_mape1 = []
            test_rmse1 = []

            test_mae2 = []
            test_mape2 = []
            test_rmse2 = []
            print('Check mask value ob1: ', mask_value1)
            print('Check mask value un: ', mask_value2)
            for i in range(self.model.horizon):
                res1 = compute_all_metrics(preds1[:,i,:], labels1[:,i,:], mask_value1)
                log1 = 'Horizon {:d}, Test MAE_ob: {:.4f}, Test RMSE_ob: {:.4f}, Test MAPE_ob: {:.4f}'
                self._logger.info(log1.format(i + 1, res1[0], res1[2], res1[1]))
                test_mae1.append(res1[0])
                test_mape1.append(res1[1])
                test_rmse1.append(res1[2])

                res2 = compute_all_metrics(preds2[:,i,:], labels2[:,i,:], mask_value2)
                log2 = 'Horizon {:d}, Test MAE_un: {:.4f}, Test RMSE_un: {:.4f}, Test MAPE_un: {:.4f}'
                self._logger.info(log2.format(i + 1, res2[0], res2[2], res2[1]))
                test_mae2.append(res2[0])
                test_mape2.append(res2[1])
                test_rmse2.append(res2[2])

                num_node_ob = len(self._node['test_observed_node'])
                num_node_un = len(self._node['test_unobserved_node'])
                mae = (res1[0] * num_node_ob + res2[0] * num_node_un) / (num_node_ob + num_node_un)
                rmse = (res1[2] * num_node_ob + res2[2] * num_node_un) / (num_node_ob + num_node_un)
                mape = (res1[1] * num_node_ob + res2[1] * num_node_un) / (num_node_ob + num_node_un)
                res2 = compute_all_metrics(preds2[:,i,:], labels2[:,i,:], mask_value2)
                log = 'Horizon {:d}, Test MAEall: {:.4f}, Test RMSEall: {:.4f}, Test MAPEall: {:.4f}'
                self._logger.info(log.format(i + 1, mae, rmse, mape))
                test_mae.append(mae)
                test_mape.append(mape)
                test_rmse.append(rmse)

            log = 'Average Test MAEall: {:.4f}, Test RMSEall: {:.4f}, Test MAPEall: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))

            log = 'Average Test MAE_ob1: {:.4f}, Test RMSE_ob: {:.4f}, Test MAPE_ob: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae1), np.mean(test_rmse1), np.mean(test_mape1)))

            log = 'Average Test MAE_un: {:.4f}, Test RMSE_un: {:.4f}, Test MAPE_un: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae2), np.mean(test_rmse2), np.mean(test_mape2)))



    def evaluate_with_ttc_n(self, mode, sample_num, step):
        if mode == 'test':
            self.load_model(self._save_path)
            self.model.eval()

        preds1, preds2, labels1, labels2 = [], [], [], []
        T = 12
        M = T // 2 + 1
        groups = 4
        node_num = len(self._node[mode + '_observed_node']) + \
                len(self._node[mode + '_unobserved_node'])
        FRP = SDC_Module(node_num, M, groups).to(self._device)
        optim = torch.optim.Adam(FRP.parameters(), lr=1e-4)

        from collections import deque
        history = deque(maxlen=T + sample_num)

        for x, label in tqdm(self._dataloader[mode + '_loader'].get_iterator()):
            # Preprocess current sample
            X = x[:, :, self._node[mode + '_node'], :]
            sem = self._sem[mode]; sem = torch.stack([sem]*X.shape[0], dim=0)
            y1 = label[:, :, self._node[mode + '_observed_node'], :]
            y2 = label[:, :, self._node[mode + '_unobserved_node'], :]
            X, y1, y2 = self._to_device(self._to_tensor([X, y1, y2]))

            # Model forward on current to collect predictions
            with torch.no_grad():
                ob_pred, un_pred, _ = self.model(X, sem)
                ob_pred = ob_pred.squeeze(0).permute(0,3,2,1)
                un_pred = un_pred.squeeze(0).permute(0,3,2,1)
                yb = torch.cat([ob_pred, un_pred], dim=2)

            FRP.eval()
            y_c = FRP(yb).permute(0,3,2,1)
            c1, c2 = torch.split(y_c, [len(self._node[mode + '_observed_node']),
                                    len(self._node[mode + '_unobserved_node'])], dim=2)
            c1, c2 = c1.unsqueeze(0), c2.unsqueeze(0)
            p1, p2, gt1, gt2 = self._inverse_transform([c1, c2, y1, y2], mode)
            preds1.append(p1.squeeze(0).squeeze(-1).cpu())
            preds2.append(p2.squeeze(0).squeeze(-1).cpu())
            labels1.append(gt1.squeeze(-1).cpu())
            labels2.append(gt2.squeeze(-1).cpu())

            # Append current for history-based training
            history.append((X, y1, y2))

            # Once enough history, update FRP with last n samples
            if len(history) == T + sample_num:
                FRP.train()
                optim.zero_grad()

                # prepare batch of n recent
                batch = list(history)[:sample_num]
                
                X_b = torch.cat([b[0] for b in batch], dim=0)  # shape [n, T, N, C]
                y1_b = torch.cat([b[1] for b in batch], dim=0)
                y2_b = torch.cat([b[2] for b in batch], dim=0)
                for _ in range(step):
                    with torch.no_grad():
                        ob_b, un_b, _ = self.model(X_b, torch.stack([sem[0]]*sample_num))
                        ob_b = ob_b.squeeze(0).permute(0,3,2,1)
                        un_b = un_b.squeeze(0).permute(0,3,2,1)
                        yb_b = torch.cat([ob_b, un_b], dim=2)

                    pred_b = FRP(yb_b).permute(0,3,2,1)
                    # combine targets
                    y_b = torch.cat([y1_b, y2_b], dim=-2)
                    pred_b, y_b = self._inverse_transform([pred_b, y_b], mode)

                    # compute loss and update
                    mask = y_b.min() if y_b.min() < 1 else 0
                    loss = masked_mae(pred_b, y_b, mask)
                    loss.backward()
                    optim.step()


        preds1 = torch.cat(preds1, dim=0)
        preds2 = torch.cat(preds2, dim=0)
        labels1 = torch.cat(labels1, dim=0)
        labels2 = torch.cat(labels2, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value1 = torch.tensor(0)
        mask_value2 = torch.tensor(0)
        if labels1.min() < 1:
            mask_value1 = labels1.min()
        if labels2.min() < 1:
            mask_value2 = labels2.min()
        if mode == 'val':
            mae1 = masked_mae(preds1, labels1, mask_value1).item()
            mape1 = masked_mape(preds1, labels1, mask_value1).item()
            rmse1 = masked_rmse(preds1, labels1, mask_value1).item()
            mae2 = masked_mae(preds2, labels2, mask_value2).item()
            mape2 = masked_mape(preds2, labels2, mask_value2).item()
            rmse2 = masked_rmse(preds2, labels2, mask_value2).item()
            loss = (mae1*labels1.shape[-1] + mae2*labels2.shape[-1])/(labels1.shape[-1]+labels2.shape[-1])
            return mae1, mape1, rmse1, mae2, mape2, rmse2, loss
        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            
            test_mae1 = []
            test_mape1 = []
            test_rmse1 = []

            test_mae2 = []
            test_mape2 = []
            test_rmse2 = []
            print('Check mask value ob1: ', mask_value1)
            print('Check mask value un: ', mask_value2)
            for i in range(self.model.horizon):
                res1 = compute_all_metrics(preds1[:,i,:], labels1[:,i,:], mask_value1)
                log1 = 'Horizon {:d}, Test MAE_ob: {:.4f}, Test RMSE_ob: {:.4f}, Test MAPE_ob: {:.4f}'
                self._logger.info(log1.format(i + 1, res1[0], res1[2], res1[1]))
                test_mae1.append(res1[0])
                test_mape1.append(res1[1])
                test_rmse1.append(res1[2])

                res2 = compute_all_metrics(preds2[:,i,:], labels2[:,i,:], mask_value2)
                log2 = 'Horizon {:d}, Test MAE_un: {:.4f}, Test RMSE_un: {:.4f}, Test MAPE_un: {:.4f}'
                self._logger.info(log2.format(i + 1, res2[0], res2[2], res2[1]))
                test_mae2.append(res2[0])
                test_mape2.append(res2[1])
                test_rmse2.append(res2[2])

                num_node_ob = len(self._node['test_observed_node'])
                num_node_un = len(self._node['test_unobserved_node'])
                mae = (res1[0] * num_node_ob + res2[0] * num_node_un) / (num_node_ob + num_node_un)
                rmse = (res1[2] * num_node_ob + res2[2] * num_node_un) / (num_node_ob + num_node_un)
                mape = (res1[1] * num_node_ob + res2[1] * num_node_un) / (num_node_ob + num_node_un)
                res2 = compute_all_metrics(preds2[:,i,:], labels2[:,i,:], mask_value2)
                log = 'Horizon {:d}, Test MAEall: {:.4f}, Test RMSEall: {:.4f}, Test MAPEall: {:.4f}'
                self._logger.info(log.format(i + 1, mae, rmse, mape))
                test_mae.append(mae)
                test_mape.append(mape)
                test_rmse.append(rmse)

            log = 'Average Test MAEall: {:.4f}, Test RMSEall: {:.4f}, Test MAPEall: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))

            log = 'Average Test MAE_ob1: {:.4f}, Test RMSE_ob: {:.4f}, Test MAPE_ob: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae1), np.mean(test_rmse1), np.mean(test_mape1)))

            log = 'Average Test MAE_un: {:.4f}, Test RMSE_un: {:.4f}, Test MAPE_un: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae2), np.mean(test_rmse2), np.mean(test_mape2)))