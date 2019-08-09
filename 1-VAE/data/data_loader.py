#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: dataloader.py
#	> Author: Yang Zhang
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Sun 04 Aug 2019 08:36:37 AM CST
# ************************************************************************/


from __future__ import print_function
import os
import os.path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# a simple dataloader example
'''
class myDataset(Dataset):
    def __init__(self, indata):
        self.data=indata
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
'''


class VoxDev_DataLoader(Dataset):

    training_file = 'training.pt'

    @property
    def train_labels(self):
        return self.targets

    @property
    def train_data(self):
        return self.data

    def __init__(self, npz_path):
        #super(VoxDev_DataLoader, self).__init__(root)
        self.training_file = npz_path


        self.vox_xvector = np.load(self.training_file)['vector']
        self.vox_utt = np.load(self.training_file)['utt']

        training_set = (self.vox_xvector, self.vox_utt)

        '''
        with open("./train_data", 'wb') as f:
            torch.save(training_set, f)
		'''

        # load pytorch tensor
        self.data, self.targets = torch.load("./train_data")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (xvector, target) where target is index of the target class.
        """
        xvector, target = self.data[index], self.targets[index]

        return xvector, target

    def __len__(self):
        return len(self.data)


class STIW_DataLoader(Dataset):
    def __init__(self, indata):
        self.data = indata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
	test = VoxDev_DataLoader("voxceleb_combined_200000/xvector.npz")
