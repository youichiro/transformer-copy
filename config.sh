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

PRETRAIN_CORPUS='bccwj_clean_char'
TRAIN_CORPUS='lang8_char'
TEST_CORPUS='naist_char'
TRAIN_PREF='lang8_char.train'
VALID_PREF='lang8_char.dev'
TEST_PREF='naist_clean_char'
DICT='dict_char.src.txt'

M2_FILE='naist_clean_char.m2'
DATA_ART=data_art/$PRETRAIN_CORPUS
DATA_BIN=$OUT/data_bin/$TRAIN_CORPUS
DATA_RAW=$OUT/data_raw/$TEST_CORPUS
mkdir -p $DATA_BIN
mkdir -p $DATA_RAW

MODELS=$OUT/models/models$exp
RESULT=$OUT/results/result$exp
