import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
import numpy as np
import configparser
from tqdm import tqdm

from models.model import PatchSTG
from lib.utils import log_string, loadData, _compute_loss, metric


class SDC_Module(nn.Module):
    """SDC_Module"""
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


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        log_string(log, '\n------------ Loading Data -------------')
        # tod and dow means the time of the day and the day of the week
        # ori_parts_idx indicates the original indices of input points
        # reo_parts_idx indicates the patched indices of input points
        # reo_all_idx indicates the patched indices of input and padded points
        self.trainX, self.trainY, self.trainXTE, self.trainYTE,\
        self.valX, self.valY, self.valXTE, self.valYTE,\
        self.testX, self.testY, self.testXTE, self.testYTE,\
        self.mean, self.std,\
        self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx = loadData(
                                        self.traffic_file, self.meta_file,
                                        self.input_len, self.output_len,
                                        self.train_ratio, self.test_ratio,
                                        self.adj_file, self.recur_times,
                                        self.tod, self.dow,
                                        self.spa_patchsize, log)
        log_string(log, '------------ End -------------\n')

        self.best_epoch = 0

        self.device = torch.device(f"cuda:{self.cuda}" if torch.cuda.is_available() else "cpu")
        self.build_model()
    
    def build_model(self):
        self.model = PatchSTG(self.tem_patchsize, self.tem_patchnum,
                            self.node_num, self.spa_patchsize, self.spa_patchnum,
                            self.tod, self.dow,
                            self.layers, self.factors,
                            self.input_dims, self.node_dims, self.tod_dims, self.dow_dims,
                            self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                        lr=self.learning_rate,weight_decay=self.weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[1,35,40],
            gamma=0.5,
        )

    def vali(self):
        self.model.eval()
        num_val = self.valX.shape[0]
        pred = []
        label = []

        num_batch = math.ceil(num_val / self.batch_size)
        with torch.no_grad():
            for batch_idx in range(num_batch):
                if isinstance(self.model, torch.nn.Module):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_val, (batch_idx + 1) * self.batch_size)

                    X = self.valX[start_idx : end_idx]
                    Y = self.valY[start_idx : end_idx]
                    TE = torch.from_numpy(self.valXTE[start_idx : end_idx]).to(self.device)
                    NormX = torch.from_numpy((X-self.mean)/self.std).float().to(self.device)

                    y_hat = self.model(NormX,TE)

                    pred.append(y_hat.cpu().numpy()*self.std+self.mean)
                    label.append(Y)
        
        pred = np.concatenate(pred, axis = 0)
        label = np.concatenate(label, axis = 0)

        maes = []
        rmses = []
        mapes = []

        for i in range(pred.shape[1]):
            mae, rmse , mape = metric(pred[:,i,:], label[:,i,:])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
        
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))
        
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)

    def train(self):
        log_string(log, "======================TRAIN MODE======================")
        min_loss = 10000000.0
        num_train = self.trainX.shape[0]

        for epoch in tqdm(range(1,self.max_epoch+1)):
            self.model.train()
            train_l_sum, train_acc_sum, batch_count, start = 0.0, 0.0, 0, time.time()
            permutation = np.random.permutation(num_train)
            self.trainX = self.trainX[permutation]
            self.trainY = self.trainY[permutation]
            self.trainXTE = self.trainXTE[permutation]
            num_batch = math.ceil(num_train / self.batch_size)
            with tqdm(total=num_batch) as pbar:
                for batch_idx in range(num_batch):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_train, (batch_idx + 1) * self.batch_size)

                    X = self.trainX[start_idx : end_idx]
                    Y = self.trainY[start_idx : end_idx]
                    
                    TE = torch.from_numpy(self.trainXTE[start_idx : end_idx]).to(self.device)
                    NormX = torch.from_numpy((X-self.mean)/self.std).float().to(self.device)
                    
                    Y = torch.from_numpy(Y).float().to(self.device)
                    
                    self.optimizer.zero_grad()

                    y_hat = self.model(NormX,TE)

                    loss = _compute_loss(Y, y_hat*self.std+self.mean)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizer.step()
                    
                    train_l_sum += loss.cpu().item()

                    batch_count += 1
                    pbar.update(1)
            log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.1f sec'
                % (epoch, self.optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
            mae, rmse, mape = self.vali()
            self.lr_scheduler.step()
            if mae[-1] < min_loss:
                self.best_epoch = epoch
                min_loss = mae[-1]
                torch.save(self.model.state_dict(), self.model_file)
        
        log_string(log, f'Best epoch is: {self.best_epoch}')

    def test(self):
        log_string(log, "======================TEST MODE======================")
        self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
        self.model.eval()
        num_val = self.testX.shape[0]
        pred = []
        label = []

        num_batch = math.ceil(num_val / self.batch_size)
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batch)):
                if isinstance(self.model, torch.nn.Module):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_val, (batch_idx + 1) * self.batch_size)

                    X = self.testX[start_idx : end_idx]
                    Y = self.testY[start_idx : end_idx]
                    TE = torch.from_numpy(self.testXTE[start_idx : end_idx]).to(self.device)
                    NormX = torch.from_numpy((X-self.mean)/self.std).float().to(self.device)
                    
                    y_hat = self.model(NormX,TE)

                    pred.append(y_hat.cpu().numpy()*self.std+self.mean)
                    label.append(Y)
        
        pred = np.concatenate(pred, axis = 0)
        label = np.concatenate(label, axis = 0)

        maes = []
        rmses = []
        mapes = []

        for i in range(pred.shape[1]):
            mae, rmse , mape = metric(pred[:,i,:], label[:,i,:])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
        
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))
        
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)
    
    def test_with_ttc(self):
        log_string(log, "======================TEST MODE======================")
        self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
        self.model.eval()
        num_val = self.testX.shape[0]
        pred = []
        label = []
        
        T = 12
        M = T // 2 + 1
        groups = 4
        FRP = SDC_Module(self.model.node_num, M, groups).to(self.device)
        optim = torch.optim.Adam(FRP.parameters(), lr=1e-4)
        
        import queue
        q = queue.Queue(maxsize=T)
        
        num_batch = math.ceil(num_val / self.batch_size)
        
        
        for batch_idx in tqdm(range(num_batch)):
            if isinstance(self.model, torch.nn.Module):
                                
                start_idx = batch_idx * self.batch_size
                end_idx = min(num_val, (batch_idx + 1) * self.batch_size)

                X = self.testX[start_idx : end_idx]
                Y = self.testY[start_idx : end_idx]
                TE = torch.from_numpy(self.testXTE[start_idx : end_idx]).to(self.device)
                NormX = torch.from_numpy((X-self.mean)/self.std).float().to(self.device)
                
                with torch.no_grad():
                    y_hat = self.model(NormX,TE)
                    y_hat = y_hat.permute(0, 3, 2, 1)
                FRP.eval()
                yc = FRP(y_hat)
                yc = yc.permute(0, 3, 2, 1)

                pred.append(yc.cpu().detach().numpy()*self.std+self.mean)
                label.append(Y)
                
                concat_x = torch.cat((NormX, TE), dim = -1)
                
                q.put((concat_x, Y))
                if q.full():
                    x_o, y_o = q.get()
                    
                    x_old, te_old = torch.split(x_o, [1, 2], dim=-1)
                    
                    y_o = torch.from_numpy(y_o).float().to(self.device)
                    
                    with torch.no_grad():
                        yb_o = self.model(x_old, te_old)
                        yb_o = yb_o.permute(0, 3, 2, 1)
                    FRP.train()
                    yc_o = FRP(yb_o)
                    yc_o = yc_o.permute(0, 3, 2, 1)
                    
                    loss = _compute_loss(y_o, yc_o*self.std+self.mean)
                    loss.backward()
                    optim.step()
                    optim.zero_grad()
                    FRP.eval()
                
        
        pred = np.concatenate(pred, axis = 0)
        label = np.concatenate(label, axis = 0)

        maes = []
        rmses = []
        mapes = []

        for i in range(pred.shape[1]):
            mae, rmse , mape = metric(pred[:,i,:], label[:,i,:])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
        
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))
        
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='configuration file')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    parser.add_argument('--cuda', type=str, default=config['train']['cuda'])
    parser.add_argument('--seed', type = int, default = config['train']['seed'])
    parser.add_argument('--batch_size', type = int, default = config['train']['batch_size'])
    parser.add_argument('--max_epoch', type = int, default = config['train']['max_epoch'])
    parser.add_argument('--learning_rate', type=float, default = config['train']['learning_rate'])
    parser.add_argument('--weight_decay', type=float, default = config['train']['weight_decay'])

    parser.add_argument('--input_len', type = int, default = config['data']['input_len'])
    parser.add_argument('--output_len', type = int, default = config['data']['output_len'])
    parser.add_argument('--train_ratio', type = float, default = config['data']['train_ratio'])
    parser.add_argument('--val_ratio', type = float, default = config['data']['val_ratio'])
    parser.add_argument('--test_ratio', type = float, default = config['data']['test_ratio'])

    parser.add_argument('--layers', type=int, default = config['param']['layers'])
    parser.add_argument('--tem_patchsize', type = int, default = config['param']['tps'])
    parser.add_argument('--tem_patchnum', type = int, default = config['param']['tpn'])
    parser.add_argument('--factors', type=int, default = config['param']['factors'])
    parser.add_argument('--recur_times', type = int, default = config['param']['recur'])
    parser.add_argument('--spa_patchsize', type = int, default = config['param']['sps'])
    parser.add_argument('--spa_patchnum', type = int, default = config['param']['spn'])
    parser.add_argument('--node_num', type = int, default = config['param']['nodes'])
    parser.add_argument('--tod', type=int, default = config['param']['tod'])
    parser.add_argument('--dow', type=int, default = config['param']['dow'])
    parser.add_argument('--input_dims', type=int, default = config['param']['id'])
    parser.add_argument('--node_dims', type=int, default = config['param']['nd'])
    parser.add_argument('--tod_dims', type=int, default = config['param']['td'])
    parser.add_argument('--dow_dims', type=int, default = config['param']['dd'])

    parser.add_argument('--traffic_file', default = config['file']['traffic'])
    parser.add_argument('--meta_file', default = config['file']['meta'])
    parser.add_argument('--adj_file', default = config['file']['adj'])
    parser.add_argument('--model_file', default = config['file']['model'])
    parser.add_argument('--log_file', default = config['file']['log'])

    args = parser.parse_args()

    log = open(args.log_file, 'w')

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    log_string(log, '------------ Options -------------')
    for k, v in vars(args).items():
        log_string(log, '%s: %s' % (str(k), str(v)))
    log_string(log, '-------------- End ----------------')

    solver = Solver(vars(args))

    # solver.train()
    solver.test()
    # solver.test_with_ttc()
    
