#!/bin/bash

## set path to dataset here
#history_sent_num_for_ks=-1
#model_for_ks=roberta-large
#dataroot=data_domain_detection_multiwoz_dstc_3way
#num_gpus=4
#
#python3 -m torch.distributed.launch --nproc_per_node ${num_gpus} --master_port 1235 baseline.py \
#    --negative_sample_method "oracle" \
#    --params_file baseline/configs/domain-detection/params.json \
#    --dataroot ${dataroot} \
#    --eval_dataset test \
#    --eval_dataroot data_eval \
#    --model_name_or_path trained_models/domain-detection \
#    --model_for_ks ${model_for_ks} \
#    --output_dir pred \
#    --exp_name domain-detection \
#    --history_sent_num_for_ks ${history_sent_num_for_ks} \
#    --eval_only \
#    --checkpoint trained_models/domain-detection \
#    --labels_file pred/ktd.test.json

# this is for generating predictions for subjective KMDM
history_sent_num_for_ks=-1
model_for_ks=roberta-large
dataroot=data_domain_detection_multiwoz_dstc_3way
num_gpus=4

python3 -m torch.distributed.launch --nproc_per_node ${num_gpus} --master_port 1235 baseline.py \
    --negative_sample_method "oracle" \
    --params_file baseline/configs/domain-detection/params.json \
    --dataroot ${dataroot} \
    --eval_dataset test \
    --eval_dataroot files_for_di_model \
    --model_name_or_path trained_models/domain-detection \
    --model_for_ks ${model_for_ks} \
    --output_dir pred \
    --exp_name domain-detection \
    --history_sent_num_for_ks ${history_sent_num_for_ks} \
    --eval_only \
    --checkpoint trained_models/domain-detection \
    --labels_file files_for_di_model/test/labels.json