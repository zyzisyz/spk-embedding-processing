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
        # super(MNIST, self).__init__(root)
        self.training_file = npz_path

        data_file = self.training_file

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )

        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)

        # load pytorch tensor
        self.data, self.targets = torch.load(data_file)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (xvector, target) where target is index of the target class.
        """
        xvector, target = self.data[index], int(self.targets[index])


        if self.target_transform is not None:
            target = self.target_transform(target)

        return xvector, target

    def __len__(self):
        return len(self.data)


class STIW_DataLoader(Dataset):
    def __init__(self, indata):
        self.data=indata
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

