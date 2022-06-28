import os
import random
import torch
import numpy as np
from dataloader import get_dataloader
from train import train
from test import test

import argparse
import warnings

warnings.filterwarnings(action='ignore')


if __name__ == "__main__":

    os.makedirs('./models/', exist_ok = True)
    os.makedirs('./result/', exist_ok = True)
    os.makedirs("result/{}".format('dagmm'), exist_ok=True)
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input_dim', type=int, default=14)
    parser.add_argument('--hidden1_dim', type=int, default=11)
    parser.add_argument('--hidden2_dim', type=int, default=9)
    parser.add_argument('--hidden3_dim', type=int, default=7)
    parser.add_argument('--zc_dim', type=int, default=1)
    
    parser.add_argument('--n_gmm', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lambda1', type=float, default=0.001)
    parser.add_argument('--lambda2', type=float, default=0.005)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--print_iter', type=int, default=50)
    parser.add_argument('--savestep_epoch', type=int, default=1)
    
    parser.add_argument(
        '--save_path', type=str, default='./models/')
    parser.add_argument('--img_dir', type=str, default='./result/')
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--data_path', type=str, default='data/anomalyDetect.csv')
    parser.add_argument('--processed_data', type=str, default='data/new_data.csv')
    parser.add_argument('--time_phase', type=bool, default=False)
    parser.add_argument('--window_size', type=int, default=1)
    parser.add_argument('--slide_size', type=int, default=1)
    parser.add_argument('--total_period', type=int, default=20)

    parser.add_argument('--ae_type', type=str, default='-')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test_all_point'])

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    

    if args.mode == 'train':
        # shuffle_dict
        shuffle_dict = {'trn': True, 'val': False, 'tst': False}
        train_loader, valid_loader, test_loader = get_dataloader(args, shuffle_dict)
        train(args, train_loader, valid_loader)

    elif args.mode == 'test_all_point':
        # shuffle_dict
        shuffle_dict = {'trn': False, 'val': False, 'tst': False}
        train_loader, valid_loader, test_loader = get_dataloader(args, shuffle_dict)
        # test(args, train_loader)
        model = load_model(args)
        test(args, model, train_loader, train_loader, save_path='result/{}/train_result.csv'.format('dagmm'))
        test(args, model, train_loader, valid_loader, save_path='result/{}/valid_result.csv'.format('dagmm'))
        test(args, model, train_loader, test_loader, save_path='result/{}/test_result.csv'.format('dagmm'))
