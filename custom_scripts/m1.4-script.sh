data_root_dir=./data/magsample
collection_path=$data_root_dir/collection/
experiment_dir=experiments/full-t5seq-aq
model_dir="./$experiment_dir/t5_docid_gen_encoder_0"
pretrained_path=$model_dir/checkpoint
index_dir=$model_dir/aq_index
mmap_dir=$model_dir/mmap
out_dir=$model_dir/aq_out

python t5_pretrainer/aq_preprocess/create_customized_smtid_file.py \
    --model_dir=$model_dir \
    --M=32 \
    --bits=8

python -m t5_pretrainer.aq_preprocss.change_embed_layer \
    --model_dir=$model_dir