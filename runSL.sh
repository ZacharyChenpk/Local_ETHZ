#!/bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export CUDA_LAUNCH_BLOCKING=1
nohup python3 -u main.py --mode train --order offset --model_path model --device 4 --method SL > train_SL.log 2>&1 &
