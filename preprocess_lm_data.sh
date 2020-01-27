#!/usr/bin/env bash

python preprocess.py \
  --trainpref data/bccwj_clean2_char.tgt \
  --validpref data/lang8_char.dev.tgt \
  --testpref data/naist_clean_char.tgt \
  --destdir out/data_bin_lm/bccwj_clean2_char \
  --output-format binary \
  --task language_modeling \
  --only-source \
  --workers 8

python /lab/ogawa/scripts/slack/send_slack_message.py -m "Finish preprocess_lm_data"

