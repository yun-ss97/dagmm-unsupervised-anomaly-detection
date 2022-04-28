import numpy as np
import os
import torch
import time
from dataloader import get_dataloader
from dataset import BuildDataset
from dagmm import DAGMM
from tqdm import tqdm
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import IPython
import argparse

# has_threshold = False

def load_model(args):
    model = DAGMM(args)
    try:
        model.load_state_dict(torch.load('./models/ngmm{}_{}.pth'.format(args.n_gmm, args.data_name[:-2])))
        print('success to load model')
    except Exception as e:
        print('Failed to load model: %s' % (e))
        exit(1)
    
    return model


def test(args, train_loader, scaler):
    model = load_model(args)
    model.eval()

    with open(os.path.join(args.data_root, args.data_name), 'rb') as f:
        print('{} is loaded'.format(args.data_name))
        df = pickle.load(f)

        df = pd.DataFrame(df)
        # apply min-max scaler fitted on train data
        df.iloc[:,:-3] = scaler.transform(df.iloc[:,:-3])

        dataset = BuildDataset(df, window_size=60, slide_size=1,time_phase=args.time_phase)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=0)


    energy_list = []
    sum_prob, sum_mean, sum_cov = 0,0,0
    data_size = 0

    with torch.no_grad():

        for input_data in train_loader:

            _ ,_, z, gamma = model(input_data)
            m_prob, m_mean, m_cov = model.get_gmm_param(gamma, z)

            sum_prob += m_prob
            sum_mean += m_mean * m_prob
            sum_cov += m_cov * m_prob
            
            data_size += input_data.shape[0]
        
        train_prob = sum_prob / data_size
        train_mean = sum_mean / sum_prob
        train_cov = sum_cov / sum_prob


        for _ ,x in enumerate(dataloader):

            _ ,_ ,z,gamma = model(x)
            
            for i in range(z.shape[0]):
                zi = z[i].unsqueeze(1)
                sample_energy = model.sample_energy(train_prob, train_mean, train_cov, zi, gamma.shape[1],gamma.shape[0])
                se = sample_energy.detach().item()
                energy_list.append(se)

    energy_list = pd.DataFrame(energy_list)
    energy_list = energy_list.replace(np.inf, 90)
    energy_list = energy_list.reset_index()
    energy_list['index'] = df.index
    energy_list = energy_list.rename(columns= {0:'anomaly_score','index':'date'})
    energy_list = energy_list.set_index('date')

    energy_list.to_csv('./result/energy_ngmm{}_{}.csv'.format(args.n_gmm, args.data_name[:-2]))
    print('energy_list saved!')