#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: model_utils.py
#	> Author: Yang Zhang
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Thu 01 Aug 2019 11:40:11 PM CST
# ************************************************************************/

from __future__ import print_function
import torch


def save_model(model, model_dir="./pth", model_name="model.pth"):
    if os.path.exists(model_path) == False:
        os.mkdir("pth")
    model_path = model_path + "/model.pth"
    torch.save(model.state_dict(), model_path)


def load_model(model, model_dir="./pth", model_name="model.pth"):
    model.load_state_dict(torch.load(path))
