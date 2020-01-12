#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
nohup python3 -u main.py --mode train --order offset --model_path model --method SL > train_SL.log 2>&1 &
