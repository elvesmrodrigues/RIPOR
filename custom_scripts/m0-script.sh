python -m torch.distributed.launch --nproc_per_node=2 -m t5_pretrainer.main \
    --epochs=50  \
    --run_name=t5_docid_gen_encoder_0 \
    --learning_rate=1e-4 \
    --loss_type=t5seq_pretrain_margin_mse \ 
    --model_name=t5-base \
    --model_type=t5_docid_gen_encoder  \
    --teacher_score_path=./data/magsample/bm25_run/qrel_added_qid_docids_teacher_scores.train.jsonl \ 
    --output_dir=./experiments/full-t5seq-aq \ 
    --task_name='["rank"]' \ 
    --collection_path=./data/magsample/collection \ 
    --max_length=512 \
    --per_device_train_batch_size=32 \
    --queries_path=./data/magsample/train_queries \ 
    --pretrained_path=t5-base \
    --use_fp16