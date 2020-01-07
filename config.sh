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

PRETRAIN_CORPUS='backtrans_bccwj_clean2_char'
TRAIN_CORPUS='lang8_uniq_char'
TRAIN_PREF='lang8_uniq_char.train'
VALID_PREF='lang8_uniq_char.dev'
DICT='dict_char.src.txt'

DATA_ART=data_art/$PRETRAIN_CORPUS
DATA_BIN=$OUT/data_bin/$TRAIN_CORPUS
mkdir -p $DATA_BIN

MODELS=$OUT/models/models$exp
RESULT=$OUT/results/result$exp

