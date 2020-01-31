#!/usr/bin/env bash
set -e
set -u

if [ $# -ne 3 ]; then
  echo "usage: python generate.sh <device> <exp> <result>" 1>&2
  exit 1
fi

DEVICE=$1
EXP=$2
RESULT=$3

# TEST_PREF='naist_clean_char'
TEST_PREF='lang8_char.dev'

EPOCH='_last'
MODEL_PATH="out/models/models${EXP}/checkpoint${EPOCH}.pt"
OPTION_FILE='option_files/exp.txt'
TEST_DATA="data/${TEST_PREF}.src"
DATA_RAW="out/data_raw/naist_clean_char"
SAVE_DIR="out/results/result${EXP}/${TEST_PREF}/${RESULT}"
SAVE_FILE="output${EPOCH}.char.txt"
OUTPUT_M2_FILE="m2score${EPOCH}.char.txt"
TRANSLM_DATA='out/models_lm/bccwj_clean2_char+lang8_char_train/checkpoint_last.pt'
TRANSLM_DICT='out/data_bin_lm/bccwj_clean2_char+lang8_char_train/dict.txt'
KENLM_DATA='/lab/ogawa/tools/kenlm/data/bccwj_clean2_char/bccwj_clean2_char.6gram.binary'
LM_WEIGHT=0.05
N_ROUND=1

mkdir -p $SAVE_DIR

if [[ -f ${SAVE_DIR}/${SAVE_FILE} ]]; then
  echo "already exists: ${SAVE_DIR}/${SAVE_FILE}" 1>&2
  exit 1
fi


CUDA_VISIBLE_DEVICES=$DEVICE python gec_model.py \
  --model-path $MODEL_PATH \
  --data-raw $DATA_RAW \
  --option-file $OPTION_FILE \
  --test-data $TEST_DATA \
  --save-dir $SAVE_DIR \
  --save-file $SAVE_FILE \
  --lm transformer_lm \
  --lm-data $TRANSLM_DATA \
  --lm-dict $TRANSLM_DICT \
  --lm-weight $LM_WEIGHT \
  --n-round $N_ROUND

echo "| calc M2score"
python2 ./gec_scripts/m2scorer/m2scorer -v ${SAVE_DIR}/${SAVE_FILE} data/${TEST_PREF}.m2 > ${SAVE_DIR}/${OUTPUT_M2_FILE}
tail -n 3 ${SAVE_DIR}/${OUTPUT_M2_FILE}

python /lab/ogawa/scripts/slack/send_slack_message.py -m "Finish gec_model: $RESULT ($HOSTNAME)"
