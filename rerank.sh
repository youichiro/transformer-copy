#!/usr/bin/env bash
set -e

# envs
ORIGIN_EXP='_lang8_uniq_unidic_with_pretrain_bccwj_clean_unidic'
RERANK_EXP='_lang8_uniq_unidic_with_pretrain_bccwj_clean_unidic'
DICT='dict_unidic.src.txt'
DEVICE=0
BEAM=12
M2FILE='naist_clean_char.m2'

NBEST_FILE="out/results/result${ORIGIN_EXP}/output_last.nbest.txt"
MODEL="out/models/models${RERANK_EXP}/checkpoint_last.pt"

tmp=out/tmp
data_raw=$tmp/data_raw
rm -rf $tmp
mkdir $tmp

mkdir out/results_rerank


# convert output{epoch}.nbest.txt to src/tgt corpus
echo "| Wrote splited $NBEST_FILE to $tmp/output.src and $tmp/output.tgt"
python gec_scripts/split_nbest.py $NBEST_FILE $tmp/output
cp $NBEST_FILE $tmp/origin${ORIGIN_EXP}_output_last.nbest.txt


# preprocess
copy_params='--copy-ext-dict'
common_params="--source-lang src --target-lang tgt
--padding-factor 1
--srcdict ./dicts/${DICT}
--joined-dictionary
"

python preprocess.py $common_params $copy_params \
  --testpref $tmp/output \
  --destdir $data_raw \
  --output-format raw

mv $data_raw/test.src-tgt.src $data_raw/test.src-tgt.src.old


# generate
python gec_scripts/split.py $data_raw/test.src-tgt.src.old $data_raw/test.src-tgt.src $data_raw/test.idx

CUDA_VISIBLE_DEVICES=$DEVICE python generate.py $data_raw \
  --path $MODEL \
  --beam 1 \
  --nbest 1 \
  --gen-subset test \
  --max-tokens 6000 \
  --no-progress-bar \
  --raw-text \
  --batch-size 128 \
  --print-alignment \
  --max-len-a 0 \
  --no-early-stop \
  --copy-ext-dict \
  --replace-unk \
  --score-hypotheses \
  > $tmp/output_rerank.nbest.txt


# rerank
python gec_scripts/sort_rerank.py $NBEST_FILE $tmp/output_rerank.nbest.txt $BEAM > $tmp/output_rerank.txt


# m2score
python gec_scripts/tokenize_character.py -f $tmp/output_rerank.txt -o $tmp/output_rerank.char.txt
python2 gec_scripts/m2scorer/m2scorer -v $tmp/output_rerank.char.txt $DATA/$M2FILE > $tmp/m2score.char.log
python gec_scripts/show_m2.py $tmp/m2score.char.log

cp -r $tmp out/results_rerank/rerank_$RERANK_EXP

# send slack message
python /lab/ogawa/scripts/slack/send_slack_message.py -m "Finish rerank: ${RERANK_EXP}"

