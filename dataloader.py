import os
import torch
from dataset import BuildDataset
import pandas as pd
from sklearn import preprocessing


def get_dataloader(args, shuffle_dict):
    df = pd.read_csv(args.processed_data)
    df = df.set_index('TIME_IDX')
    
    trn = df[df['data']=='trn']
    val = df[df['data']=='val']
    tst = df[df['data']=='tst']
    
    min_max_scaler = preprocessing.MinMaxScaler()
    trn.iloc[:, :-4] = min_max_scaler.fit_transform(trn.iloc[:, :-4].values)
    val.iloc[:, :-4] = min_max_scaler.transform(val.iloc[:, :-4].values)
    tst.iloc[:, :-4] = min_max_scaler.transform(tst.iloc[:, :-4].values)

    trn_dataset = BuildDataset(args, trn, args.window_size, args.slide_size)
    val_dataset = BuildDataset(args, val, args.window_size, args.slide_size)
    tst_dataset = BuildDataset(args, tst, args.window_size, slide_size=1)

    trn_dataloader = torch.utils.data.DataLoader(
        trn_dataset, batch_size=args.batch_size, shuffle=shuffle_dict['trn'], num_workers=8, drop_last=False
        )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=shuffle_dict['val'], num_workers=8, drop_last=False
    )
    tst_dataloader = torch.utils.data.DataLoader(
        tst_dataset, batch_size=args.batch_size, shuffle=shuffle_dict['tst'], num_workers=8, drop_last=False
    )

    return trn_dataloader, val_dataloader, tst_dataloader

