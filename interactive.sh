#!/usr/bin/env bash
source ./config.sh

copy_params='--copy-ext-dict --replace-unk'
if $WO_COPY; then
    copy_params='--replace-unk'
fi

beam=12

CUDA_VISIBLE_DEVICES=1 python interactive.py $DATA_RAW \
--path ./out/models/models_lang8_unidic_reverse_without_pretrain/checkpoint11.pt \
--beam $beam \
--nbest $beam \
--no-progress-bar \
--print-alignment \
$copy_params

#--replace-unk ./data/bin/alignment.src-tgt.txt \
#--path $MODELS/checkpointema1.pt \
