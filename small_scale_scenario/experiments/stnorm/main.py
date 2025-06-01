import os
import argparse
import numpy as np

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.stnorm import STNORM
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
    parser.add_argument('--model_name', type=str, default='stnorm')
    parser.add_argument('--tnorm_bool', type=bool, default=True)
    parser.add_argument('--snorm_bool', type=bool, default=True)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--blocks', type=int, default=4)
    parser.add_argument('--layers', type=int, default=2)

    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
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

    model = STNORM(node_num=node_num,
                input_dim=args.input_dim,
                output_dim=args.output_dim,
                tnorm_bool=args.tnorm_bool,
                snorm_bool=args.snorm_bool,
                channels=args.channels,
                kernel_size=args.kernel_size,
                blocks=args.blocks,
                layers=args.layers,
                )
    
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = None

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