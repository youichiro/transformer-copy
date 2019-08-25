#!/usr/bin/env bash

device=0
if [ $# -ge 1 ]; then
    device=$1
fi

exp=''
if [ $# -ge 2 ]; then
    exp=$2
fi

DATA='data_ja'
OUT='out_ja'
CORPUS_PREF='bccwj'
TRAIN_PREF='lang8_unidic.train'
VALID_PREF='lang8_unidic.dev'
TEST_PREF='naist_clean_unidic'
DICT='dict_ja.src.txt'
DATA_ART="data_ja_art/$CORPUS_PREF"

DATA_BIN=$OUT/data_bin
DATA_RAW=$OUT/data_raw
mkdir -p $DATA_BIN
mkdir -p $DATA_RAW

MODELS=$OUT/models$exp
RESULT=$OUT/result$exp
# mkdir -p $MODELS
# mkdir -p $RESULT
