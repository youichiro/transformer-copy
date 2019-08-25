source ./config.sh
mkdir data_align

# trainpref='data/train_merge'
# trainpref='data/valid'
trainpref=$DATA/$TRAIN_PREF
fast_align_dir='/lab/ogawa/tools/fast_align/build/'
mosesdecoder_dir='/lab/ogawa/tools/mosesdecoder/'

python scripts/build_sym_alignment.py \
--fast_align_dir $fast_align \
--mosesdecoder_dir $mosesdecoder_dir \
--source_file $trainpref.src \
--target_file $trainpref.tgt \
--output_dir data_align

cp data_align/align.forward $trainpref.forward
cp data_align/align.backward $trainpref.backward

rm -rf data_align
