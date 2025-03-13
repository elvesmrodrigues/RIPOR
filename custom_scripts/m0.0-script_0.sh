data_root_dir=./data/ripor
collection_path=$data_root_dir/collection/
experiment_dir=experiments/full-t5seq-aq
model_dir="./$experiment_dir/t5_docid_gen_encoder_0"
pretrained_path=$model_dir/checkpoint
index_dir=$model_dir/index
out_dir=$model_dir/out

python -m torch.distributed.launch --nproc_per_node=1 -m t5_pretrainer.evaluate \
    --pretrained_path=$pretrained_path \
    --index_dir=$index_dir \
    --out_dir=$out_dir \
    --task=index \
    --encoder_type=t5seq_pretrain_encoder \
    --collection_path=$collection_path
