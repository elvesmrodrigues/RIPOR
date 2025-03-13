data_root_dir=./data/magsample
collection_path=$data_root_dir/collection/
experiment_dir=experiments/full-t5seq-aq
model_dir="./$experiment_dir/t5_docid_gen_encoder_0"
pretrained_path=$model_dir/checkpoint
index_dir=$model_dir/aq_index
mmap_dir=$model_dir/mmap
out_dir=$model_dir/aq_out

python -m t5_pretrainer.evaluate \
--task=mmap_2 \
--index_dir=$mmap_dir \
--mmap_dir=$mmap_dir
