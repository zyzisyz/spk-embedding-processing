#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: network.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Thu 01 Aug 2019 07:19:51 PM CST
# ************************************************************************/


from __future__ import print_function
import torch
from torch import nn, optim
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.g1 = nn.Linear(784, 1000)
        self.g2 = nn.Linear(1000, 1000)
        self.g3 = nn.Linear(1000, 1000)
        self.g_out = nn.Linear(1000, 40)

        self.f1 = nn.Linear(20, 1000)
        self.f2 = nn.Linear(1000, 784)

    def encode(self, x):
        x = F.relu(self.g1(x))
        x = F.relu(self.g2(x))
        x = F.relu(self.g3(x))
        x = self.g_out(x) 
        mean = x[:,:20]
        sigma = F.softplus(x[:,20:])
        return mean, sigma

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x = F.relu(self.f1(z))
        return torch.sigmoid(self.f2(x))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

