#!/usr/bin/env bash

device=0
if [ $# -ge 1 ]; then
  device=$1
fi

exp=''
if [ $# -ge 2 ]; then
  exp=$2
fi

DATA='data'
OUT='out'


TEST_CORPUS='naist_unidic'
TEST_PREF='naist_clean_unidic'

M2_FILE='naist_clean_char.m2'
DATA_RAW=$OUT/data_raw/$TEST_CORPUS

# MODELS=$OUT/models/models$exp
MODELS=(\
  $OUT/models/models_lang8_without_pretrain/checkpoint_last.pt \
  $OUT/models/models_lang8_without_pretrain/checkpoint_best.pt \
)
MODELS=(`echo $MODELS|tr ' ', ':'`)
echo "models $MODELS"
RESULT=$OUT/results/result$exp
