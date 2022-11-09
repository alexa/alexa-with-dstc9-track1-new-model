#!/bin/bash

## set path to dataset here
#history_sent_num_for_ks=1
#model_for_ks=roberta-large
#dataroot=data_ktd
#eval_dataroot=data_eval
#eval_dataset=test
#num_gpus=4
#OUT_DIR=pred
#
#mkdir -p ${OUT_DIR}
#
##CUDA_VISIBLE_DEVICES=${gpu_max}
#python3 -m torch.distributed.launch --master_port 1233 --nproc_per_node ${num_gpus} baseline.py \
#        --params_file baseline/configs/detection/params.json \
#        --model_name_or_path trained_models/ktd \
#        --model_for_ks ${model_for_ks} \
#        --history_sent_num_for_ks ${history_sent_num_for_ks} \
#        --dataroot ${dataroot} \
#        --exp_name ktd-${version} \
#        --eval_dataroot ${eval_dataroot} \
#        --eval_dataset ${eval_dataset} \
#        --output_dir ${OUT_DIR} \
#        --output_file ${OUT_DIR}/ktd.${eval_dataset}.json \
#        --eval_desc ${eval_dataset} \
#        --seed 42 \
#        --eval_only \
#        --checkpoint trained_models/ktd


# this is for generating predictions for subjective KMDM
history_sent_num_for_ks=1
model_for_ks=roberta-large
dataroot=data_ktd
eval_dataroot=files_for_di_model
eval_dataset=test
num_gpus=4
OUT_DIR=pred

mkdir -p ${OUT_DIR}

#CUDA_VISIBLE_DEVICES=${gpu_max}
python3 -m torch.distributed.launch --master_port 1233 --nproc_per_node ${num_gpus} baseline.py \
        --params_file baseline/configs/detection/params.json \
        --model_name_or_path trained_models/ktd \
        --model_for_ks ${model_for_ks} \
        --history_sent_num_for_ks ${history_sent_num_for_ks} \
        --dataroot ${dataroot} \
        --exp_name ktd-${version} \
        --eval_dataroot ${eval_dataroot} \
        --eval_dataset ${eval_dataset} \
        --output_dir ${OUT_DIR} \
        --output_file ${OUT_DIR}/ktd.${eval_dataset}.json \
        --eval_desc ${eval_dataset} \
        --seed 42 \
        --eval_only \
        --checkpoint trained_models/ktd