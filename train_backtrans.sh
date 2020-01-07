#!/usr/bin/env bash
set -x

device=0
exp='_backtrans_lang8_char'
TRAIN_CORPUS='backtrans_lang8_char'
MODELS="out/models/models${exp}"
DATA_BIN="out/data_bin/${TRAIN_CORPUS}"

mkdir $MODELS

CUDA_VISIBLE_DEVICES=$device python train.py $DATA_BIN \
  --save-dir $MODELS \
  --seed 4321 \
  --max-epoch 30 \
  --batch-size 32 \
  --max-tokens 3000 \
  --train-subset train \
  --valid-subset valid \
  --arch transformer \
  --lr-scheduler triangular \
  --max-lr 0.004 --lr-period-updates 73328 \
  --clip-norm 2 --lr 0.001 \
  --lr-shrink 0.95 --shrink-min \
  --dropout 0.2 --relu-dropout 0.2 \
  --attention-dropout 0.2 \
  --encoder-embed-dim 512 --decoder-embed-dim 512 \
  --max-target-positions 1024 --max-source-positions 1024 \
  --encoder-ffn-embed-dim 4096 --decoder-ffn-embed-dim 4096 \
  --encoder-attention-heads 8 --decoder-attention-heads 8 \
  --copy-attention \
  --copy-attention-heads 1 \
  --copy-attention-dropout 0.2 \
  --no-progress-bar \
  --log-interval 1000 \
  --share-all-embeddings \
  --weight-decay 0.0 \
  --no-ema \
  --positive-label-weight 3.0 \
  --fix-batches-to-gpus \
  | tee out/log/log${exp}.out

python /lab/ogawa/scripts/slack/send_slack_message.py -m "Finish trainig: ${exp}"

