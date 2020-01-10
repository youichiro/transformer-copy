#!/usr/bin/env bash

DATA_RAW=out/data_raw/naist_clean_char
MODEL=out/models/models_lang8_char_with_pretrain_ja_bccwj_clean_char_2/checkpoint_last.pt
beam=12

python interactive.py $DATA_RAW \
  --path $MODEL \
  --beam $beam \
  --nbest $beam \
  --no-progress-bar \
  --print-alignment \
  --copy-ext-dict \
  --replace-unk \
  --cpu

