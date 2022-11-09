#TRAIN_BS=1
#EVAL_BS=8
#model=pegasus-large
#OUTPUT_DIR=trained_models/generation
#eval_dataroot=data_eval
#eval_dataset=test
#
## convert ks predictions into source sentences for generation
#python prepare-generation-input.py ${eval_dataroot} ${eval_dataset} pred/ks.test.json pred
#
## start generation
#python generation-finetune.py \
#--data_dir=pred \
#--model_name_or_path ${OUTPUT_DIR}/best_tfmr \
#--task summarization \
#--learning_rate=2e-5 \
#--train_batch_size=${TRAIN_BS} \
#--eval_batch_size=${EVAL_BS} \
#--eval_beams 5 \
#--output_dir=${OUTPUT_DIR} \
#--max_source_length=256 \
#--max_target_length=64 \
#--val_max_target_length 64 \
#--test_max_target_length 64 \
#--val_check_interval=1.0 \
#--val_metric rouge2 \
#--do_predict \
#--gpus 1 \
#--num_train_epochs 5 \
##--check_output_dir \
##--do_train \
#
#python scripts/scores_pipeline.py \
#--outfile ${OUTPUT_DIR}/test_generations.txt \
#--reffile ${eval_dataroot}/${eval_dataset}/labels.json \
#--ksfile pred/ks.test.json \
#--finalfile pred/${model}-response-final.json \
#--scorefile pred/${model}-scores.json
#
#cat pred/${model}-scores.json

# this is for generating predictions for subjective KMDM
TRAIN_BS=1
EVAL_BS=4
#model=pegasus-large
model=t5-large
OUTPUT_DIR=trained_models/generation/${model}
eval_dataroot=files_for_di_model
#eval_dataroot=data_eval
eval_dataset=test

# convert ks predictions into source sentences for generation
python prepare-generation-input.py ${eval_dataroot} ${eval_dataset} files_for_di_model/gt_ktd.gt_ks.json pred
#python prepare-generation-input.py ${eval_dataroot} ${eval_dataset} pred/ks.test.json pred

# start generation
python generation-finetune.py \
--data_dir=pred \
--model_name_or_path ${OUTPUT_DIR}/best_tfmr \
--task summarization \
--learning_rate=2e-5 \
--train_batch_size=${TRAIN_BS} \
--eval_batch_size=${EVAL_BS} \
--eval_beams 5 \
--output_dir=${OUTPUT_DIR} \
--max_source_length=256 \
--max_target_length=64 \
--val_max_target_length 64 \
--test_max_target_length 64 \
--val_check_interval=1.0 \
--val_metric rouge2 \
--do_predict \
--gpus 1 \
--num_train_epochs 5 \

python scripts/scores_pipeline.py \
--outfile ${OUTPUT_DIR}/test_generations.txt \
--reffile ${eval_dataroot}/${eval_dataset}/labels.json \
--finalfile pred/${model}-response-t5.json \
--scorefile pred/${model}-scores.json \
--ksfile files_for_di_model/gt_ktd.gt_ks.json \
#--ksfile pred/ks.test.json \

cat pred/${model}-scores.json