data_root_dir=./data/ripor
collection_path=$data_root_dir/collection/
experiment_dir=experiments/full-t5seq-aq
model_dir="./$experiment_dir/t5_docid_gen_encoder_0"
pretrained_path=$model_dir/checkpoint
index_dir=$model_dir/index
out_dir=$model_dir/out

python -m t5_pretrainer.evaluate \
--task=index_2 \
--index_dir=$index_dir 