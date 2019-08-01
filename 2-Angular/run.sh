#!/bin/bash

#*************************************************************************
#	> File Name: run.sh
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com 
#	> Created Time: Thu 01 Aug 2019 06:25:08 PM CST
# ************************************************************************/


python -c main.py --batch-size 64 \
				  --epochs 10 \
				  --lr 0.01 \
				  --save-model True
