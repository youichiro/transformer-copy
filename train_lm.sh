#!/usr/bin/env bash
set -e
set -u

device=$1

# memory watch
# memory_watch_running=`ps aux|grep ogawa|grep memory_watch.sh|grep -v color|grep -v grep`
# if [ -z "$memory_watch_running" ]; then
#   sh /lab/ogawa/scripts/server/memory_watch.sh &
#   mpid=$!
#   echo "| pid of memory_watch: ${mpid}"
# else
#   mpid=0
#   echo "| memory_watch.sh is running"
# fi


DATA_BIN=out/data_bin_lm/bccwj_clean2_char+lang8_char_train
MODELS=out/models_lm/bccwj_clean2_char+lang8_char_train


CUDA_VISIBLE_DEVICES=$device python train.py $DATA_BIN \
  --task language_modeling \
  --save-dir $MODELS \
  --arch transformer_lm \
  --share-decoder-input-output-embed \
  --dropout 0.2 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --min-lr 1e-08 \
  --tokens-per-sample 1024 --sample-break-mode none \
  --max-tokens 8192 --update-freq 16 \
  --max-update 50000 \
  --no-ema \
  --no-progress-bar \
  --log-interval 10 \

# finish
# python /lab/ogawa/scripts/slack/send_slack_message.py -m "Finish trainig LM"
# if [ $mpid -ne 0 ]; then
#   kill -9 $mpid
# fi

