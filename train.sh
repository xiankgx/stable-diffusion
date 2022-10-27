#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3 python main.py --base configs/stable-diffusion/v1-finetune.yaml -t --gpus 0,1,2