import argparse

def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--dataset', type=str, default='PEMS08')
    parser.add_argument('--fewshot', type=int, default=5, help='create few-shot percent datasets, 100 means full-shot')
    parser.add_argument('--eval_method', type=str, default='norm')
    parser.add_argument('--use_long', type=bool, default=False)
    
    # if need to use the data from multiple years, please use underline to separate them, e.g., 2018_2019
    parser.add_argument('--years', type=str, default='2016')
    parser.add_argument('--seed', type=int, default=2016)

    parser.add_argument('--bs', type=int, default=16)
    # seq_len denotes input history length, horizon denotes output future length
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=1)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    return parser