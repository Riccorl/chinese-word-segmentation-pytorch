#!/bin/bash
source /home/ric/miniconda3/bin/activate pt

scripts/train_evaluate.sh pku | tee logs/lstm/log_sgd_pku.txt
scripts/train_evaluate.sh msr | tee logs/lstm/log_sgd_msr.txt
scripts/train_evaluate.sh cityu | tee logs/lstm/log_sgd_cityu.txt
scripts/train_evaluate.sh as | tee logs/lstm/log_sgd_as.txt
