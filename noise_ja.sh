#!/usr/bin/env bash

set -e
set -u
set -x

corpus='data/bccwj_clean_unidic.chunk'
plabels='data/bccwj_clean_unidic.plabels'
pset='data/pset.txt'
data_art='data_art/ja_bccwj_clean_char'

mkdir -p $data_art

python noise_ja.py -c $corpus -l $plabels -p $plabels -o $data_art -e 1 -s 9182
python noise_ja.py -c $corpus -l $plabels -p $plabels -o $data_art -e 2 -s 78834
python noise_ja.py -c $corpus -l $plabels -p $plabels -o $data_art -e 3 -s 5101
python noise_ja.py -c $corpus -l $plabels -p $plabels -o $data_art -e 4 -s 33302
python noise_ja.py -c $corpus -l $plabels -p $plabels -o $data_art -e 5 -s 781
python noise_ja.py -c $corpus -l $plabels -p $plabels -o $data_art -e 6 -s 1092
python noise_ja.py -c $corpus -l $plabels -p $plabels -o $data_art -e 7 -s 10688
python noise_ja.py -c $corpus -l $plabels -p $plabels -o $data_art -e 8 -s 50245
python noise_ja.py -c $corpus -l $plabels -p $plabels -o $data_art -e 9 -s 71187

python /lab/ogawa/scripts/slack/send_slack_message.py -m "Finish noise_ja"