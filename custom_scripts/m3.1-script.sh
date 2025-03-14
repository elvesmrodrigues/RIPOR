task=t5seq_aq_get_qid_to_smtid_rankdata

experiment_dir=experiments/full-t5seq-aq
data_dir="./$experiment_dir/t5_docid_gen_encoder_0"
data_dir="./$experiment_dir/t5_docid_gen_encoder_0"
docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json 

model_dir=./$experiment_dir/t5seq_aq_encoder_seq2seq_1
pretrained_path=$model_dir/checkpoint
train_query_dir="./data/magsample/train_queries/"

# Since prefix=32 and prefix=16 almost corresponds to the same doc. To save time, we directly expand it from 16 to 32.
python t5_pretrainer/aq_preprocess/expand_smtid_for_qid_smtid_docids.py \
    --data_dir=$data_dir \
    --src_qid_smtid_rankdata_path=$model_dir/sub_smtid_16_out/qid_smtid_rankdata.json \
    --out_dir=$model_dir/sub_smtid_32_out

python t5_pretrainer/aq_preprocess/argparse_from_qid_smtid_rank_to_qid_smtid_docids.py \
        --root_dir=$model_dir/sub_smtid_32_out

# let's rerank the data
echo "Reranking the data not implemented yet" 
# for max_new_token in 4 8 16 32
# do 
#     qid_smtid_docids_path=$model_dir/sub_smtid_"$max_new_token"_out/qid_smtid_docids.train.json

#     python -m torch.distributed.launch --nproc_per_node=2 -m t5_pretrainer.rerank \
#         --train_queries_path=$train_queries_path \
#         --collection_path=$collection_path \
#         --model_name_or_path=cross-encoder/ms-marco-MiniLM-L-6-v2 \
#         --max_length=256 \
#         --batch_size=256 \
#         --qid_smtid_docids_path=$qid_smtid_docids_path \
#         --task=cross_encoder_rerank_for_qid_smtid_docids

#     python -m t5_pretrainer.rerank \
#         --out_dir=$model_dir/sub_smtid_"$max_new_token"_out \
#         --task=cross_encoder_rerank_for_qid_smtid_docids_2
# done