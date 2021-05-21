import os
import random
import torch
import numpy as np
from dataloader import get_dataloader
from train import train
from test import test
from visualization import make_plot

import argparse
import warnings

warnings.filterwarnings(action='ignore')


if __name__ == "__main__":

    os.makedirs('./models/', exist_ok = True)
    os.makedirs('./result/', exist_ok = True)
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input_dim', type=int, default=31)
    parser.add_argument('--hidden1_dim', type=int, default=23)
    parser.add_argument('--hidden2_dim', type=int, default=17)
    parser.add_argument('--hidden3_dim', type=int, default=11)
    parser.add_argument('--zc_dim', type=int, default=5)
    
    parser.add_argument('--n_gmm', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lambda1', type=float, default=0.1)
    parser.add_argument('--lambda2', type=float, default=0.005)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--print_iter', type=int, default=300)
    parser.add_argument('--savestep_epoch', type=int, default=2)
    
    parser.add_argument('--save_path', type=str, default='./models/')
    parser.add_argument('--img_dir', type=str, default='./result/')
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--data_root', type=str, default='/mnt/C83AFA0C3AF9F6F2/21_hanwha_1/DC101_preprocessed_sample')
    parser.add_argument('--data_name', type=str, default='20180125_20180125.p')
    parser.add_argument('--time_phase', type=bool, default=False)

    parser.add_argument('--mode', type=str, choices=['train', 'test', 'plot_result'])

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_loader, valid_loader, _, scaler = get_dataloader(args)

    print('Experiment {} is ongoing!'.format(args.data_name))

    if args.mode == 'train':
        train(args, train_loader, valid_loader)

    elif args.mode == 'test':
        test(args, train_loader, scaler)
        