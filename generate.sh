#!/usr/bin/env bash
set -e
set -u

if [ $# -ne 2 ]; then
  echo "usage: python generate.sh <device> <exp>" 1>&2
  exit 1
fi

device=$1
exp=$2

MODELS=out/models/models${exp}
# data_raws=('naist_clean_char')
data_raws=('lang8_char.dev')
epochs=('_last' '_best')

for data_raw in ${data_raws[*]}; do
  for epoch in ${epochs[*]}; do
  # for epoch in {1..30}; do
      echo -e "\n${data_raw} ${epoch}"
      RESULT=out/results/result${exp}/${data_raw}
      output_pref=$RESULT/output${epoch}
      output_m2score=$RESULT/m2score${epoch}.char.log
      mkdir -p $RESULT

      CUDA_VISIBLE_DEVICES=$device python generate.py out/data_raw/${data_raw} \
      --path $MODELS/checkpoint$epoch.pt \
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
      > ${output_pref}.nbest.txt

      if [ $? -gt 0 ]; then
        python /lab/ogawa/scripts/slack/send_slack_message.py -m "Error!! generate: ${exp} ($HOSTNAME)"
        exit 1
      fi

      cat ${output_pref}.nbest.txt | grep "^H" | python ./gec_scripts/sort.py 12 ${output_pref}.txt
      python ./gec_scripts/tokenize_character.py -f ${output_pref}.txt -o ${output_pref}.char.txt
      python2 ./gec_scripts/m2scorer/m2scorer -v ${output_pref}.char.txt data/${data_raw}.m2 > $output_m2score
      echo "[m2score]"
      tail -n 3 $output_m2score
  done
done

python /lab/ogawa/scripts/slack/send_slack_message.py -m "Finish generate: ${exp} ($HOSTNAME)"

