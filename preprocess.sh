#!/usr/bin/env bash
source ./config.sh

set -e
set -x

rm -rf $DATA_BIN
rm -rf $DATA_RAW

# set copy params
copy_params='--copy-ext-dict'

# set common params between train/test
common_params="--source-lang src --target-lang tgt
--padding-factor 1
--srcdict ./dicts/${DICT}
--joined-dictionary
"

trainpref=$DATA/$TRAIN_PREF
validpref=$DATA/$VALID_PREF
testpref=$DATA/$TEST_PREF

# preprocess train/valid
python preprocess.py \
$common_params \
$copy_params \
--trainpref $trainpref \
--validpref $validpref \
--destdir $DATA_BIN \
--output-format binary \
--alignfile $trainpref.forward | tee $OUT/log/data_bin.log

# preprocess test
python preprocess.py \
$common_params \
$copy_params \
--testpref $testpref \
--destdir $DATA_RAW \
--output-format raw | tee $OUT/log/data_raw.log

mv $DATA_RAW/test.src-tgt.src $DATA_RAW/test.src-tgt.src.old
