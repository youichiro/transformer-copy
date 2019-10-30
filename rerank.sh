#!/usr/bin/env bash
set -e

# envs
NBEST_FILE='out/results/result_lang8_char_with_pretrain_bccwj_char/output_last.nbest.txt'
DICT='dict_unidic.src.txt'

tmp=out/tmp
data_raw=$tmp/data_raw
rm -rf $tmp
mkdir $tmp


# convert output{epoch}.nbest.txt to src/tgt corpus
echo "| Wrote splited $NBEST_FILE to $tmp/output.src and $tmp/output.tgt"
python ./gec_scripts/split_nbest.py $NBEST_FILE $tmp/output


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




