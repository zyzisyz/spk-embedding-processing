#!/bin/bash

#*************************************************************************
#	> File Name: train.sh
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com 
#	> Created Time: Thu 01 Aug 2019 06:19:03 PM CST
# ************************************************************************/


python -u main.py --batch-size 128 \
	--epochs 200 

