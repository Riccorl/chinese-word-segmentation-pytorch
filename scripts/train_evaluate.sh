#!/bin/bash
source /home/pollo/miniconda3/bin/activate pt-gpu

# args needed
# input file "/home/ric/external/datadrive/Datasets/chinese-word-seg/processed/training/pku_training.utf8"
# test file "/home/ric/external/datadrive/Datasets/chinese-word-seg/processed/testing/pku_test.utf8"
# gold file "/home/ric/external/datadrive/Datasets/chinese-word-seg/processed/gold/pku_test_gold.utf8"
# dict file for score "/home/ric/external/datadrive/Datasets/chinese-word-seg/processed/gold/pku_training_words.utf8"
# language model
#   - clue/roberta_chinese_clue_tiny
#   - adamlin/bert-distil-chinese
#   - best-base-chinese
#   - voidful/albert_chinese_base
#   - voidful/albert_chinese_small
#   - voidful/albert_chinese_tiny
#   - hfl/chinese-roberta-wwm-ext

# variables
BASE_PATH="/mnt/d/Datasets/chinese-word-seg/processed"
INPUT_FILE="$BASE_PATH/training/$1_training.utf8"
TEST_FILE="$BASE_PATH/testing/$1_test.utf8"
GOLD_FILE="$BASE_PATH/gold/$1_test_gold.utf8"
DICT_FILE="$BASE_PATH/gold/$1_training_words.utf8"
PREDICT_FILE="predictions/$1_pred.utf8"
EMB="resources/bigram_unigram300.bin"
LM=$2
EPOCHS=60
BATCH_SIZE=64
N_LAYER=1
HIDDEN_SIZE=256
# train
if [ -z "$2" ]; then
    # default value, not used if not present.
    # prevents python script crash
    LM="bert-base-chinese"
fi
python cws/train.py \
    --input_file "$INPUT_FILE" \
    --batch_size $BATCH_SIZE \
    --max_epochs $EPOCHS \
    --hidden_size $HIDDEN_SIZE \
    --num_layer $N_LAYER \
    --bert_mode concat \
    --gpus 1 \
    --run 30 \
    --language_model $LM \
    --dataset "$1" \
    --optimizer "adamw" \
    --gradient_clip_val 0.5 \
    --optimized_decay \
    --lr 0.001 \
    # --schedule \
    # --freeze \
    # --embeddings_file $EMB \

sleep 1
BEST_MODEL_PATH=$(cat predictions/best_model_path.txt)
echo "Best model path: $BEST_MODEL_PATH"
# predict
python cws/predictor.py "$TEST_FILE" "$PREDICT_FILE" "$BEST_MODEL_PATH"
# evaluate
if [ "$(wc -l <"$PREDICT_FILE")" -eq "$(wc -l <"$GOLD_FILE")" ]; then
    echo 'Same number of lines, ok'
    echo "scripts/score $DICT_FILE $GOLD_FILE $PREDICT_FILE"
    scripts/score "$DICT_FILE" "$GOLD_FILE" "$PREDICT_FILE"
else
    echo 'Number of lines mismatch!'
fi
