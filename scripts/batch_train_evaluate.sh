#!/bin/bash
source /home/ric/miniconda3/bin/activate pt

# scripts/train_evaluate.sh pku voidful/albert_chinese_tiny | tee logs/albert/log_pku.txt
scripts/train_evaluate.sh msr | tee logs/bert/log_msr.txt
scripts/train_evaluate.sh cityu | tee logs/bert/log_cityu.txt
scripts/train_evaluate.sh as | tee logs/bert/log_as.txt
