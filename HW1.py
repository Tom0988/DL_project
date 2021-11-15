# PyTorch
from abc import ABC

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# data preprocess model
import numpy as np
import csv
import os

# plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# some function
from function import get_device, plot_learning_curve, plot_pred


class Covid19Dataset(Dataset):
    def __init__(self, path, mode='train', target_only=False):
        self.mode = mode

        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)

        if not target_only:  # 只會過base line
            feats = list(range(93))
        else:  # 做出來能過medium line
            pass

        if mode == 'test':
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            target = data[:, -1]
            data = data[:, feats]

            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]

            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(data[indices])

        # normalize features
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)


def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):

    dataset = Covid19Dataset(path, mode=mode, target_only=target_only)  # Construct dataset
    dataloader = DataLoader(  # Construct dataloader
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)
    return dataloader



