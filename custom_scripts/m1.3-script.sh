data_root_dir=./data/magsample
collection_path=$data_root_dir/collection/
experiment_dir=experiments/full-t5seq-aq
model_dir="./$experiment_dir/t5_docid_gen_encoder_0"
pretrained_path=$model_dir/checkpoint
index_dir=$model_dir/aq_index
mmap_dir=$model_dir/mmap
out_dir=$model_dir/aq_out

python -m t5_pretrainer.evaluate \
--task=aq_evaluate \
--pretrained_path=$pretrained_path \
--index_dir=$index_dir \
--out_dir=$out_dir \
--q_collection_paths=$q_collection_paths \
--eval_qrel_path=$eval_qrel_path  \
--mmap_dir=$mmap_dir