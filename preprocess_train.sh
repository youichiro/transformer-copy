#!/usr/bin/env bash
set -e
set -x

TRAIN_CORPUS='lang8_reverse_char'
TRAIN_PREF='lang8_reverse_char.train'
VALID_PREF='lang8_reverse_char.dev'
DICT='dict_char.src.txt'
DATA_BIN=out/data_bin/$TRAIN_CORPUS


copy_params='--copy-ext-dict'
common_params="--source-lang src --target-lang tgt
--padding-factor 1
--srcdict ./dicts/${DICT}
--joined-dictionary
"

# preprocess train/valid
python preprocess.py $common_params $copy_params \
  --trainpref data/$TRAIN_PREF \
  --validpref data/$VALID_PREF \
  --destdir $DATA_BIN \
  --output-format binary \
  --alignfile data/${TRAIN_PREF}.forward \
  | tee out/log/data_bin.log

