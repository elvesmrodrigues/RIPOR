data_root_dir=./data/magsample
collection_path=$data_root_dir/collection/
experiment_dir=experiments/full-t5seq-aq

task=t5seq_aq_encoder_seq2seq
query_to_docid_path=data/magsample/absTtitle/title_to_docid.train.json
data_dir="./$experiment_dir/t5_docid_gen_encoder_0"
docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json
output_dir="./$experiment_dir/"

model_dir="./$experiment_dir/t5_docid_gen_encoder_0"
pretrained_path=$model_dir/no_share_checkpoint/
run_name=t5seq_aq_encoder_seq2seq_0

python -m torch.distributed.launch --nproc_per_node=2 -m t5_pretrainer.main \
    --max_steps=250_000 \
    --run_name=$run_name  \
    --learning_rate=1e-3 \
    --loss_type=$task \
    --model_name_or_path=t5-base \
    --model_type=t5_docid_gen_encoder \
    --per_device_train_batch_size=256 \
    --pretrained_path=$pretrained_path \
    --query_to_docid_path=$query_to_docid_path \
    --docid_to_smtid_path=$docid_to_smtid_path \
    --output_dir=$output_dir \
    --save_steps=50_000 \
    --task_names='["rank"]' \
    --wandb_project_name=full_t5seq_encoder \
    --use_fp16 \
    --warmup_ratio=0.045