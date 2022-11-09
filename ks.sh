#!/bin/bash

# set path to dataset here
knowledge_selection_method=qa+entity
history_sent_num_for_ks=-1
model_for_ks=roberta-large
train_dataroot=data
n_candidates=5
negative_sample_method=mix
negative_mix_percent='0.0,0.1,0.1,0.7'
use_tfidf_rate=0.0
num_gpus=1

# training
#python3 -m torch.distributed.launch --nproc_per_node ${num_gpus} --master_port 1232 baseline.py \
#    --negative_sample_method ${negative_sample_method} \
#    --params_file baseline/configs/selection/params.json \
#    --dataroot ${train_dataroot} \
#    --model_name_or_path /home/ec2-user/models/${model_for_ks} \
#    --model_for_ks ${model_for_ks} \
#    --exp_name ks-all \
#    --knowledge_selection_method ${knowledge_selection_method} \
#    --history_sent_num_for_ks ${history_sent_num_for_ks} \
#    --negative_mix_percent ${negative_mix_percent} \
#    --n_candidates ${n_candidates} \
#    --use_tfidf_rate ${use_tfidf_rate} \
#    --use_hinge_loss \
##

## test set eval
#eval_dataset=test
#eval_dataroot=data_eval
#entities_file=pred/entities_detected.${eval_dataset}.final.json
#labels_file=pred/ktd.${eval_dataset}.json
#OUT_DIR=pred
#mkdir -p ${OUT_DIR}
#
#python3 baseline.py --eval_only --checkpoint trained_models/ks \
#   --dataroot {eval_dataroot} \
#   --eval_dataroot ${eval_dataroot} \
#   --eval_dataset ${eval_dataset} \
#   --output_dir ${OUT_DIR} \
#   --output_file ${OUT_DIR}/ks.${eval_dataset}.json \
#   --model_for_ks ${model_for_ks} \
#   --knowledge_selection_method ${knowledge_selection_method} \
#   --history_sent_num_for_ks ${history_sent_num_for_ks} \
#   --entities_file ${entities_file} \
#   --labels_file ${labels_file}
##
### scoring
#python scripts/scores-ks.py --dataset ${eval_dataset} --dataroot ${eval_dataroot}/ \
#--outfile ${OUT_DIR}/ks.${eval_dataset}.json --scorefile ${OUT_DIR}/ks.${eval_dataset}.score.json
#
#cat ${OUT_DIR}/ks.${eval_dataset}.score.json


# this is for generating predictions for subjective KMDM
eval_dataset=test
eval_dataroot=files_for_di_model
entities_file=pred/entities_detected.${eval_dataset}.final.json
#labels_file=pred/ktd.${eval_dataset}.json
labels_file=files_for_di_model/test/labels.json
OUT_DIR=pred
mkdir -p ${OUT_DIR}

python3 baseline.py --eval_only --checkpoint trained_models/ks \
   --dataroot {eval_dataroot} \
   --eval_dataroot ${eval_dataroot} \
   --eval_dataset ${eval_dataset} \
   --output_dir ${OUT_DIR} \
   --output_file ${OUT_DIR}/ks.${eval_dataset}.json \
   --model_for_ks ${model_for_ks} \
   --knowledge_selection_method ${knowledge_selection_method} \
   --history_sent_num_for_ks ${history_sent_num_for_ks} \
   --entities_file ${entities_file} \
   --labels_file ${labels_file}
#
## scoring
python scripts/scores-ks.py --dataset ${eval_dataset} --dataroot ${eval_dataroot}/ \
--outfile ${OUT_DIR}/ks.${eval_dataset}.json --scorefile ${OUT_DIR}/ks.${eval_dataset}.score.json

cat ${OUT_DIR}/ks.${eval_dataset}.score.json
