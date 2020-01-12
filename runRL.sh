#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
nohup python3 -u main.py --mode train --order offset --model_path model_RL --method RL > train_RL.log &
