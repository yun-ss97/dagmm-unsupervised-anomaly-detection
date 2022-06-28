import torch
from torch.utils.data import Dataset

class BuildDataset(Dataset):
    def __init__(self, args, data, window_size, slide_size, time_phase=True):
        self.args = args
        self.data = data
        self.window_size = window_size
        self.slide_size = slide_size
        self.time_phase = time_phase
        self.start_point = []
        self.time_idx = []
        period_start = 0
        for i in range(self.args.total_period):
            temp_df = self.data[(self.data['p_idx']==i)]
            self.start_point.extend(range(period_start, period_start + len(temp_df)-self.window_size))
            self.time_idx.extend(range(period_start + self.window_size - 1, period_start + len(temp_df)-1, self.slide_size))
            period_start += len(temp_df)
        
        self.label = self.data.iloc[:,-3].values
        self.data = self.data.iloc[:,:-4].values
        self.time_idx = data.index.values[self.time_idx]

    def __len__(self):
        if self.time_phase:
            return len(self.start_point)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.time_phase:
            output = torch.FloatTensor(self.data[self.start_point[idx]:self.start_point[idx]+self.window_size]) # (batch_size x input_size x seq_len)
            if self.args.ae_type == 'mlp':
                output = output.view(([self.args.num_var * self.args.window_size]))
                
            target = self.label[self.start_point[idx]:self.start_point[idx]+self.window_size]
            if 1 in target:
                target = 1
            else:
                target = 0
            return torch.FloatTensor(output), target
        else:
            return torch.FloatTensor(self.data[idx])