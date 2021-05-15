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

# def compute_threshold(model, train_loader,len_data):
#     energies = np.zeros(shape=(len_data))
#     step = 0
#     energy_interval = 50

#     with torch.no_grad():
#         model.eval()

#         for x in train_loader:
#             enc,dec,z,gamma = model(x)
#             m_prob, m_mean, m_cov = model.get_gmm_param(gamma, z)
            
#             for i in range(z.shape[0]):
#                 zi = z[i].unsqueeze(1)
#                 sample_energy = model.sample_energy(m_prob, m_mean, m_cov, zi,gamma.shape[1], gamma.shape[0])

#                 energies[step] = sample_energy.detach().item()
#                 step += 1

#             if step % energy_interval == 0:
#                 print('Iteration: %d    sample energy: %.4f' % (step, sample_energy))
    
#     threshold = np.percentile(energies, 80)
#     print('threshold: %.4f' %(threshold))
    
#     return threshold

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
    x_list = []
    dec_list = []

    with torch.no_grad():

        for input_data in train_loader:

            enc,dec,z,gamma = model(input_data)
            m_prob, m_mean, m_cov = model.get_gmm_param(gamma, z)
            input_data,dec,z,gamma = input_data.cpu(),dec.cpu(),z.cpu(),gamma.cpu()


        for x in dataloader:
            enc,dec,z,gamma = model(x)
            
            for i in range(z.shape[0]):
                zi = z[i].unsqueeze(1)
                sample_energy = model.sample_energy(m_prob, m_mean, m_cov, zi,gamma.shape[1],gamma.shape[0])
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