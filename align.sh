source ./config.sh
mkdir data_align

trainpref=$DATA/$TRAIN_PREF
validpref=$DATA/$VALID_PREF
fast_align_dir='/lab/ogawa/tools/fast_align/build/'
mosesdecoder_dir='/lab/ogawa/tools/mosesdecoder/'

# train
python scripts/build_sym_alignment.py \
--fast_align_dir $fast_align_dir \
--mosesdecoder_dir $mosesdecoder_dir \
--source_file $trainpref.src \
--target_file $trainpref.tgt \
--output_dir data_align

cp data_align/align.forward $trainpref.forward
cp data_align/align.backward $trainpref.backward

# valid
python scripts/build_sym_alignment.py \
--fast_align_dir $fast_align_dir \
--mosesdecoder_dir $mosesdecoder_dir \
--source_file $validpref.src \
--target_file $validpref.tgt \
--output_dir data_align

cp data_align/align.forward $validpref.forward
cp data_align/align.backward $validpref.backward

rm -rf data_align
