task=t5seq_aq_encoder_margin_mse
data_root_dir=./data/magsample
collection_path=$data_root_dir/collection/
queries_path=./data/magsample/train_queries/

experiment_dir=experiments/full-t5seq-aq

data_dir="./$experiment_dir/t5_docid_gen_encoder_0"
docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json
output_dir="./$experiment_dir/"

model_dir="./$experiment_dir/t5seq_aq_encoder_seq2seq_0"
pretrained_path=$model_dir/checkpoint
run_name=t5seq_aq_encoder_seq2seq_1

teacher_score_path=./data/magsample/bm25_run/qrel_added_qid_docids_teacher_scores.train.jsonl

python -m torch.distributed.launch --nproc_per_node=2 -m t5_pretrainer.main \
    --epochs=150 \
    --run_name=$run_name \
    --learning_rate=1e-4 \
    --loss_type=t5seq_aq_encoder_margin_mse \
    --model_name_or_path=t5-base \
    --model_type=t5_docid_gen_encoder \
    --teacher_score_path=$teacher_score_path \
    --output_dir=$output_dir \
    --task_names='["rank"]' \
    --wandb_project_name=full_t5seq_encoder \
    --use_fp16 \
    --collection_path=$collection_path \
    --max_length=64 \
    --per_device_train_batch_size=128 \
    --queries_path=$queries_path \
    --pretrained_path=$pretrained_path \
    --docid_to_smtid_path=$docid_to_smtid_path
