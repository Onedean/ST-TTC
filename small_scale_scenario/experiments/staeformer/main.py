import os
import argparse
import numpy as np

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.staeformer import STAEformer
from src.base.engine import BaseEngine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, get_dataset_info
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument('--model_name', type=str, default='staeformer')
    parser.add_argument('--input_embedding_dim', type=int, default=24)
    parser.add_argument('--tod_embedding_dim', type=int, default=24)
    parser.add_argument('--dow_embedding_dim', type=int, default=24)
    parser.add_argument('--spatial_embedding_dim', type=int, default=0)
    parser.add_argument('--adaptive_embedding_dim', type=int, default=80)
    parser.add_argument('--feed_forward_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    
    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_mixed_proj', type=bool, default=True)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    if args.fewshot!=100:
        log_dir = log_dir + f'-fewshot{args.fewshot}'
    else:
        log_dir = log_dir + f'-fulltrain'
    
    if args.bs != 1:
        logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    else:
        folder_name = 'few-shot' if args.fewshot != 100 else 'full-shot'
        if args.dataset == 'CA' or args.dataset == 'GLA' or args.dataset == 'GBA':
            folder_name = 'large'
        if args.seq_len != 12:
            folder_name = 'long_term'
        logger = get_logger('./log/{}/{}/{}/'.format(folder_name, args.dataset, args.model_name), __name__, '{}_record_s{}.log'.format(args.eval_method, args.seed))
    
    logger.info(args)
    
    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)
    
    data_path, _, node_num = get_dataset_info(args.dataset)
    
    dataloader, scaler = load_dataset(data_path, args, logger)
    
    if args.dataset == 'URBANEV':  # UrbanEV (1h)
        T_O_D = 24
    elif args.dataset == 'CA' or args.dataset == 'GLA' or args.dataset == 'GBA':  # LargeST (15min)
        T_O_D = 4 * 24
    elif args.dataset == 'KNOWAIR':  # KnowAir (3h)
        T_O_D = 8
    else:
        T_O_D = 12 * 24

    model = STAEformer(node_num=node_num,
                    input_dim=args.input_dim,
                    output_dim=args.output_dim,
                    seq_len=args.seq_len,
                    horizon=args.horizon,
                    steps_per_day=T_O_D,
                    input_embedding_dim=args.input_embedding_dim,
                    tod_embedding_dim=args.tod_embedding_dim,
                    dow_embedding_dim=args.dow_embedding_dim,
                    spatial_embedding_dim=args.spatial_embedding_dim,
                    adaptive_embedding_dim=args.adaptive_embedding_dim,
                    feed_forward_dim=args.feed_forward_dim,
                    num_heads=args.num_heads,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    use_mixed_proj=args.use_mixed_proj,
                )
    
    loss_fn = masked_mae
    wdecay_config = {'PEMS03':0.0005, 'PEMS04':0.0005, 'PEMS07':0.001, 'PEMS08':0.0015, 'PEMSBAY':0.0001, 'METRLA':0.0003, "KNOWAIR":0.0015, "URBANEV":0.0015}
    wdecay = wdecay_config[args.dataset]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=wdecay)
    steps_config = {'PEMS03':[15, 30, 40], 'PEMS04':[15, 30, 50], 'PEMS07':[15, 35, 50], 'PEMS08':[25, 45, 65], 'PEMSBAY':[10, 30], 'METRLA':[20,30], 'KNOWAIR':[25, 45, 65], 'URBANEV':[25, 45, 65]}
    steps = steps_config[args.dataset]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.1, verbose=True)

    engine = BaseEngine(device=device,
                        model=model,
                        dataloader=dataloader,
                        scaler=scaler,
                        sampler=None,
                        loss_fn=loss_fn,
                        lrate=args.lrate,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        clip_grad_value=args.clip_grad_value,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        log_dir=log_dir,
                        logger=logger,
                        eval_method=args.eval_method,
                        seed=args.seed
                        )

    if args.mode == 'train':
        engine.train()
    else:
        # engine.evaluate_with_norm(args.mode)
        if args.eval_method == 'norm':
            engine.evaluate_with_norm(args.mode)
        elif args.eval_method == 'ttc':
            engine.evaluate_with_ttc(args.mode)


if __name__ == "__main__":
    main()