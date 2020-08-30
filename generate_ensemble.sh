#!/usr/bin/env bash
set -e
set -u

if [ $# -ne 2 ]; then
  echo "usage: bash generate.sh <device> <exp>" 1>&2
  exit 1
fi

device=$1
exp=$2

MODELS=(\
  out/models/models_lang8_char_with_pretrain_ja_bccwj_clean_char_2/checkpoint_last.pt \
  out/models/models_lang8_char_with_pretrain_backtrans_bccwj_clean2_char/checkpoint_last.pt \
)
MODELS=(`echo ${MODELS[@]}|tr ' ', ':'`)
echo "| models: $MODELS"

data_raws=('naist_clean_char')


for data_raw in ${data_raws[*]}; do
  echo -e "\n${data_raw}"
  RESULT=out/results/result_ensemble${exp}/${data_raw}
  output_pref=$RESULT/output_ensemble
  output_m2score=$RESULT/m2score_ensemble.char.log
  mkdir -p $RESULT
  echo $MODELS > $RESULT/models.txt

  CUDA_VISIBLE_DEVICES=$device python generate.py out/data_raw/${data_raw} \
  --path $MODELS \
  --beam 12 \
  --nbest 12 \
  --gen-subset test \
  --max-tokens 6000 \
  --batch-size 8 \
  --no-progress-bar \
  --raw-text \
  --print-alignment \
  --max-len-a 0 \
  --no-early-stop \
  --copy-ext-dict --replace-unk \
  > ${output_pref}.nbest.txt

  cat ${output_pref}.nbest.txt | grep "^H" | python ./gec_scripts/sort.py 12 ${output_pref}.txt
  python ./gec_scripts/tokenize_character.py -f ${output_pref}.txt -o ${output_pref}.char.txt
  python2 ./gec_scripts/m2scorer/m2scorer -v ${output_pref}.char.txt data/${data_raw}.m2 > $output_m2score
  echo "[m2score]"
  tail -n 3 $output_m2score

done


python /lab/ogawa/scripts/slack/send_slack_message.py -m "Finish generate_ensemble: ${exp}"

