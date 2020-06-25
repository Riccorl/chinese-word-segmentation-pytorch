#!/bin/bash
source /home/ric/miniconda3/bin/activate pt

scripts/train_evaluate.sh pku 2>&1 | tee logs/log_pku.txt
scripts/train_evaluate.sh msr 2>&1 | tee logs/log_msr.txt
scripts/train_evaluate.sh cityu 2>&1 | tee logs/log_cityu.txt
scripts/train_evaluate.sh as 2>&1 | tee logs/log_as.txt
