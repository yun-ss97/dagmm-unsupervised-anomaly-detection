import os
import pickle
import torch
from dataset import BuildDataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import IPython

def get_dataloader(args, window_size=60, slide_size=30, time_phase=False):
    '''
    Return data loader

    Parameters:
        data_root (str): path of the data
        data_name (str): name of the data
        batch_size (int): batch size
        window_size (int): window size for time series condition
        slide_size (int): moving window size
        time_phase (bool): i.i.d. if False

    Returns:
        train/dev/tst dataloader (torch.utils.data.DataLoader):
            output shape - time series: (batch size, window size, 31)
                           i.i.d: (batch size, 31)
    '''
    with open(os.path.join(args.data_root, args.data_name), 'rb') as f:
        df = pickle.load(f)


    if time_phase:
        # time phase condition
        print("time series condition")
        trn = df[df['time_phase'] == 'trn']
        dev = df[df['time_phase'] == 'val']
        tst = df
    else:
        # i.i.d. condition
        print("i.i.d. condition")
        trn = df[df['iid_phase'] == 'trn']
        dev = df[df['iid_phase'] == 'val']
        tst = df
        # IPython.embed();exit(1);
        trn = pd.DataFrame(trn)
        scaler = MinMaxScaler().fit(trn.iloc[:,:-3])
        trn.iloc[:,:-3] = scaler.transform(trn.iloc[:,:-3])
        # IPython.embed();exit(1);
        dev.iloc[:,:-3] = scaler.transform(dev.iloc[:,:-3])
        tst.iloc[:,:-3] = scaler.transform(tst.iloc[:,:-3])


    print('train #:', trn.shape)
    print('valid #:', dev.shape)
    print('test #:', tst.shape)

    trn_dataset = BuildDataset(trn, window_size, slide_size, time_phase)
    dev_dataset = BuildDataset(dev, window_size, slide_size, time_phase)
    tst_dataset = BuildDataset(tst, window_size, 1, time_phase, test=True)

    trn_dataloader = torch.utils.data.DataLoader(
        trn_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
    )
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True
    )
    tst_dataloader = torch.utils.data.DataLoader(
        tst_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True
    )

    return trn_dataloader, dev_dataloader, tst_dataloader, scaler
