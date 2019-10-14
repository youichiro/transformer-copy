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

PRETRAIN_CORPUS='bccwj_unidic'
TRAIN_CORPUS='lang8_unidic'
TEST_CORPUS='naist_unidic'
TRAIN_PREF='lang8_unidic.train'
VALID_PREF='lang8_unidic.dev'
TEST_PREF='naist_clean_unidic'
DICT='dict_unidic.src.txt'

M2_FILE='naist_clean_char.m2'
DATA_ART=data_art/$PRETRAIN_CORPUS
DATA_BIN=$OUT/data_bin/$TRAIN_CORPUS
DATA_RAW=$OUT/data_raw/$TEST_CORPUS
mkdir -p $DATA_BIN
mkdir -p $DATA_RAW

MODELS=$OUT/models/models$exp
RESULT=$OUT/results/result$exp

