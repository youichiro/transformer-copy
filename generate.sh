#!/usr/bin/env bash
source ./config.sh

set -e

ema=''
epochs=('_last' '_best')

mkdir -p $RESULT

# rm $DATA_RAW/test.src-tgt.src
python gec_scripts/split.py $DATA_RAW/test.src-tgt.src.old $DATA_RAW/test.src-tgt.src $DATA_RAW/test.idx

for epoch in ${epochs[*]}; do
    if [ -f $RESULT/m2score$ema$exp_$epoch.log ] && [ -f $RESULT/m2score$ema$exp_$epoch.char.log ]; then
        continue
    fi
    echo $epoch

    CUDA_VISIBLE_DEVICES=$device python generate.py $DATA_RAW \
    --path $MODELS/checkpoint$ema$epoch.pt \
    --beam 12 \
    --nbest 12 \
    --gen-subset test \
    --max-tokens 6000 \
    --no-progress-bar \
    --raw-text \
    --batch-size 32 \
    --print-alignment \
    --max-len-a 0 \
    --no-early-stop \
    --copy-ext-dict --replace-unk \
    > $RESULT/output$ema$epoch.nbest.txt

    cat $RESULT/output$ema$epoch.nbest.txt | grep "^H" | python ./gec_scripts/sort.py 12 $RESULT/output$ema$epoch.txt.split
    python ./gec_scripts/revert_split.py $RESULT/output$ema$epoch.txt.split $DATA_RAW/test.idx > $RESULT/output$ema$epoch.txt

    # 文字分割でM2スコアを計算する
    python ./gec_scripts/tokenize_character.py -f $RESULT/output$ema$epoch.txt -o $RESULT/output$ema$epoch.char.txt
    python2 ./gec_scripts/m2scorer/m2scorer -v $RESULT/output$ema$epoch.char.txt $DATA/$M2_FILE > $RESULT/m2score$ema$epoch.char.log

    tail $RESULT/m2score$ema$epoch.char.log

done

python /lab/ogawa/scripts/slack/send_slack_message.py -m "Finish generate: ${exp}"

