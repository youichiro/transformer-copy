#!/usr/bin/env bash
set -e
set -u

if [ $# -ne 2 ]; then
  echo "usage: python generate.sh <device> <exp>" 1>&2
  exit 1
fi

device=$1
exp=$2

DATA_RAW='naist_clean_char'

EPOCH='_last'
MODEL_PATH="out/models/models${EXP}/checkpoint${EPOCH}.pt"
OPTION_FILE='option_files/exp.txt'
DATA_RAW="out/data_raw/${DATA_RAW}"
TEST_DATA="data/${DATA_RAW}.src"
SAVE_DIR="out/results/result${EXP}/${DATA_RAW}"
SAVE_FILE="output_gecmodel${EPOCH}.char.txt"
OUTPUT_M2_FILE="m2score_gecmodel${EPOCH}.char.txt"
KENLM_DATA='/lab/ogawa/tools/kenlm/data/bccwj_clean2_char/bccwj_clean2_char.4gram.binary'
KENLM_WEIGHT=0.5


CUDA_VISIBLE_DEVICES=$device python gec_model.py \
  --model-path $MODEL_PATH \
  --data-raw $DATA_RAW \
  --option-file $OPTION_FILE \
  --test-data $TEST_DATA \
  --save-dir $SAVE_DIR \
  --save-file $SAVE_FILE \
  --kenlm-data $KENLM_DATA \
  --kenlm-weight $KENLM_WEIGHT

python2 ./gec_scripts/m2scorer/m2scorer -v ${SAVE_DIR}/${SAVE_FILE} data/${DATA_RAW}.m2 > ${SAVE_DIR}/${OUTPUT_M2_FILE}
tail -n 3 ${SAVE_DIR}/${OUTPUT_M2_FILE}
