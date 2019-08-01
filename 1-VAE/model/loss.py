#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: loss.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Thu 01 Aug 2019 07:03:51 PM CST
# ************************************************************************/

from __future__ import print_function
import torch
from torch import nn, optim
from torch.nn import functional as F


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


