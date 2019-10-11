#!/usr/bin/env bash
source ./config_ensemble.sh

set -e

mkdir -p $RESULT

rm -rf $DATA_RAW/test.src-tgt.src $DATA_RAW/test.src-tgt.tgt
python gec_scripts/split.py $DATA_RAW/test.src-tgt.src.old $DATA_RAW/test.src-tgt.src $DATA_RAW/test.idx
cp $DATA_RAW/test.src-tgt.src $DATA_RAW/test.src-tgt.tgt

if [ -f $RESULT/m2score$exp.log ] && [ -f $RESULT/m2score$exp.char.log ]; then
  continue
fi

CUDA_VISIBLE_DVICES=$device python generatte.py $DATA_RAW \
  --path $MODELS \
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
  > $RESULT/output.nbest.txt

cat $RESULT/output.nbest.txt | grep "^H" | python ./gec_scripts/sort.py 12 $RESULT/output.txt.split

python ./gec_scripts/revert_split.py $RESULT/output.txt.split $DATA_RAW/test.idx > $RESULT/output.txt

python2 ./gec_scripts/m2scorer/m2scorer -v $RESULT/output.txt $DATA/$TEST_PREF.m2 > $RESULT/m2score.log
tail -n 1 $RESULT/m2score.log

python ./gec_scripts/tokenize_character.py -f $RESULT/output.txt -o $RESULT/output.char.txt
python2 ./gec_scripts/m2scorer/m2scorer -v $RESULT/output.char.txt $DATA/$M2_FILE > $RESULT/m2score.char.log
tail -n 1 $RESULT/m2score.char.log

python gec_scripts/show_m2.py $RESULT/m2score.log
python gec_scripts/show_m2.py $RESULT/m2score.char.log

python /lab/ogawa/scripts/slack/send_slack_message.py -m "Finish generate_ensemble: ${exp}"
