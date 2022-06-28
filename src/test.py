import numpy as np
import os
import torch
import time
from dataloader import get_dataloader
from dataset import BuildDataset
from dagmm import DAGMM
import pandas as pd
import pickle


def load_model(args):
    model = DAGMM(args)
    try:
        model.load_state_dict(torch.load('./models/ngmm{}_{}.pth'.format(args.n_gmm, args.data_name[:-2])))
        print('success to load model')
    except Exception as e:
        print('Failed to load model: %s' % (e))
        exit(1)
    
    return model


def test(args, model, train_loader, target_loader, save_path):
    model.eval()
    energy_list = []

    sum_prob, sum_mean, sum_cov = 0,0,0
    data_size = 0

    with torch.no_grad():
        for input_data,_ in train_loader:
            input_data = input_data.squeeze(1)
            _ ,_, z, gamma = model(input_data)
            m_prob, m_mean, m_cov = model.get_gmm_param(gamma, z)
            sum_prob += m_prob
            sum_mean += m_mean * m_prob.unsqueeze(1)
            sum_cov += m_cov * m_prob.unsqueeze(1).unsqueeze(1)
            
            data_size += input_data.shape[0]
        
        train_prob = sum_prob / data_size
        train_mean = sum_mean / sum_prob.unsqueeze(1)
        train_cov = sum_cov / m_prob.unsqueeze(1).unsqueeze(1)

    
        for _, (x, _) in enumerate(target_loader):
            x = x.squeeze(1)
            _ ,_ ,z,gamma = model(x)
            
            for i in range(z.shape[0]):
                zi = z[i].unsqueeze(1)
                sample_energy = model.sample_energy(train_prob, train_mean, train_cov, zi, gamma.shape[1],gamma.shape[0])
                se = sample_energy.detach().item()
                energy_list.append(se)

    energy_list = pd.DataFrame(energy_list)
    energy_list = energy_list.replace(np.inf, max(energy_list))
    energy_list['date'] = list(target_loader.dataset.time_idx)
    energy_list = energy_list.rename(columns= {0:'anomaly_score'})
    energy_list.to_csv(save_path, index=False)
    print('energy_list saved!')