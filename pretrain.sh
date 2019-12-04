#!/usr/bin/env bash
source ./config.sh

DATA_BIN=out/data_bin_art/$PRETRAIN_CORPUS/data_bin_art

data_epoch=2
echo "data_epoch: ${data_epoch}"
CUDA_VISIBLE_DEVICES=0,1 python train.py ${DATA_BIN}_${data_epoch} \
  --save-dir $MODELS \
  --max-epoch $data_epoch \
  --batch-size 8 \
  --max-tokens 3000 \
  --train-subset train \
  --valid-subset valid \
  --arch transformer \
  --clip-norm 2 \
  --lr 0.002 \
  --min-lr 1e-4 \
  --lr-shrink 0.999 \
  --weight-decay 0.5 \
  --validate-interval 10 \
  --dropout 0.2 \
  --relu-dropout 0.2 \
  --attention-dropout 0.2 \
  --copy-attention-dropout 0.2 \
  --encoder-embed-dim 512 \
  --decoder-embed-dim 512 \
  --max-target-positions 1024 \
  --max-source-positions 1024 \
  --encoder-ffn-embed-dim 4096 \
  --decoder-ffn-embed-dim 4096 \
  --encoder-attention-heads 8 \
  --decoder-attention-heads 8 \
  --copy-attention-heads 1 \
  --share-all-embeddings \
  --no-progress-bar \
  --log-interval 1000 \
  --no-ema \
  --skip-invalid-size-inputs-valid-test \
  --copy-attention \
  --copy-attention-heads 1 \
  --positive-label-weight 3.0 \
  | tee $OUT/log/log$exp_${data_epoch}.out

  # --token-labeling-loss-weight 0.1 \
  # --token-labeling-positive-label-weight 5.0 \

python /lab/ogawa/scripts/slack/send_slack_message.py -m "Finish pretraining: [$data_epoch] $PRETRAIN_CORPUS"
