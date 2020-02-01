#!/usr/bin/env bash
set -e
set -x

TEST_CORPUS='lang8_char.dev'
TEST_PREF='lang8_char.dev'
DICT='dict_char.src.txt'
DATA_RAW=out/data_raw/$TEST_CORPUS
TEST_PREF=data/$TEST_PREF

copy_params='--copy-ext-dict'
common_params="--source-lang src --target-lang tgt
--padding-factor 1
--srcdict ./dicts/${DICT}
--joined-dictionary
"

# preprocess test
python preprocess.py $common_params $copy_params \
  --testpref $TEST_PREF \
  --destdir $DATA_RAW \
  --output-format raw \
  | tee out/log/data_raw.log

