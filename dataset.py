import torch
from torch.utils.data import Dataset

class BuildDataset(Dataset):
    """
    A class to build a custom dataset.

    ...

    Attributes
    ----------
    data: dataframe
        input data
    window_size: int
        window size for time phase condition
    slide_size: int
        sliding size
    time_phase: bool
        i.i.d. condition if False

    Methods
    -------
    __len__():
        Return data size
    __getitem__():
        Return stacked tensor (input)
    """
    def __init__(self, data, window_size, slide_size, time_phase=True, test=False):
        self.data = data.iloc[:, :-3].values
        if test:
            self.start_point = range(0, len(self.data)-window_size, slide_size)
        else:
            self.start_point = range(0, len(self.data), slide_size)
        self.window_size = window_size
        self.time_phase = time_phase

    def __len__(self):
        if self.time_phase:
            return len(self.start_point)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.time_phase:
            return torch.FloatTensor(self.data[self.start_point[idx]:self.start_point[idx]+self.window_size])
        else:
            return torch.FloatTensor(self.data[idx])

