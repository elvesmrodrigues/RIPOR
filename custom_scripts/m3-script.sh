data_dir="./$experiment_dir/t5_docid_gen_encoder_0"
data_dir="./$experiment_dir/t5_docid_gen_encoder_0"
docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json 

model_dir=./$experiment_dir/t5seq_aq_encoder_seq2seq_1
pretrained_path=$model_dir/checkpoint
train_query_dir="./data/magsample/train_queries/"

# Apply beam search to generate prefix with length 4, 8, 16
for max_new_token in 4 8 16
do
    out_dir=$model_dir/sub_smtid_"${max_new_token}"_out/
    python -m torch.distributed.launch --nproc_per_node=2 -m t5_pretrainer.evaluate \
        --pretrained_path=$pretrained_path \
        --out_dir=$out_dir \
        --task=$task \
        --docid_to_smtid_path=$docid_to_smtid_path \
        --topk=100 \
        --batch_size=4 \
        --train_query_dir=$train_query_dir \
        --max_new_token=$max_new_token

    python -m t5_pretrainer.evaluate \
        --task="$task"_2 \
        --out_dir=$out_dir 

    python t5_pretrainer/aq_preprocess/argparse_from_qid_smtid_rank_to_qid_smtid_docids.py \
        --root_dir=$out_dir
done