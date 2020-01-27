#!/usr/bin/env bash

python preprocess.py \
  --trainpref data/bccwj_clean2_char.tgt \
  --destdir out/data_bin_lm/bccwj_clean2_char \
  --output-format binary \
  --task language_modeling \
  --only-source \
