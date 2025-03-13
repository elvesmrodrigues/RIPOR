data_root_dir=./data/ripor
collection_path=$data_root_dir/collection/
experiment_dir=experiments/full-t5seq-aq
model_dir="./$experiment_dir/t5_docid_gen_encoder_0"
pretrained_path=$model_dir/checkpoint
index_dir=$model_dir/index
out_dir=$model_dir/out
q_collection_paths=./data/ripor/queries/

python -m torch.distributed.launch --nproc_per_node=2 -m t5_pretrainer.rerank \
    --task=rerank_for_create_trainset \
    --run_json_path=$run_path \
    --out_dir=$out_dir \
    --collection_path=$collection_path \
    --q_collection_path=$q_collection_path \
    --json_type=json \
    --batch_size=256

python -m t5_pretrainer.rerank \
    --task=rerank_for_create_trainset_2 \
    --out_dir=$out_dir
