#!/bin/bash
source /home/ric/miniconda3/bin/activate pt

# args needed
# input file "/home/ric/external/datadrive/Datasets/chinese-word-seg/processed/training/pku_training.utf8"
# test file "/home/ric/external/datadrive/Datasets/chinese-word-seg/processed/testing/pku_test.utf8"
# gold file "/home/ric/external/datadrive/Datasets/chinese-word-seg/processed/gold/pku_test_gold.utf8"
# dict file for score "/home/ric/external/datadrive/Datasets/chinese-word-seg/processed/gold/pku_training_words.utf8"
# language model
#   - "clue/roberta_chinese_clue_tiny"
#   - "adamlin/bert-distil-chinese"
#   - "best-base-chinese"

# variables
BASE_PATH="/home/ric/external/datadrive/Datasets/chinese-word-seg/processed"
INPUT_FILE="$BASE_PATH/training/$1_training.utf8"
TEST_FILE="$BASE_PATH/testing/$1_test.utf8"
GOLD_FILE="$BASE_PATH/gold/$1_test_gold.utf8"
DICT_FILE="$BASE_PATH/gold/$1_training_words.utf8"
PREDICT_FILE="predictions/$1_pred.utf8"
LM=$2
EPOCHS=30
BATCH_SIZE=32
N_LAYER=1
HIDDEN_SIZE=256
MAX_LEN=120
# train
python cws/train.py \
    --input_file $INPUT_FILE \
    --batch_size $BATCH_SIZE \
    --max_epochs $EPOCHS \
    --hidden_size $HIDDEN_SIZE \
    --num_layer $N_LAYER \
    --max_len $MAX_LEN \
    --language_model $LM \
    --gradient_clip_val 0.5 \
    --bert_mode sum \
    --gpus 1

sleep 1
BEST_MODEL_PATH=$(cat predictions/best_model_path.txt)
echo $BEST_MODEL_PATH
# predict
python cws/predictor.py $TEST_FILE $PREDICT_FILE $BEST_MODEL_PATH
# evaluate
if [ "$(wc -l <$PREDICT_FILE)" -eq "$(wc -l <$GOLD_FILE)" ]; then
    echo 'Same number of lines, ok'
    scripts/score $DICT_FILE $GOLD_FILE $PREDICT_FILE
else
    echo 'Number of lines mismatch!'
fi
