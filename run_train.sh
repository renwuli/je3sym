#!/bin/bash

log_silent=1 CUDA_VISIBLE_DEVICES=0 python train.py --config configs/train.yaml
