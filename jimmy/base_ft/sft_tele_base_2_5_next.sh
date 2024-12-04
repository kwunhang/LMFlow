#!/bin/bash
# Please run this script under ${project_id} in project directory of

# Parses arguments
model_name_or_path=/ssddata/jimmy/model/Qwen2.5-14B
lora_model_path=/ssddata/jimmy/ft-model/sft-Qwen2.5-14B
dataset_path=/ssddata/cug-llm-data/sft_teleQNA
conversation_template=qwen
output_dir=/ssddata/jimmy/ft-model/sft-tele-Qwen2.5-14B
deepspeed_args="--master_port=11000"

# Safety related arguments
trust_remote_code=0

while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -m|--model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    -d|--dataset_path)
      dataset_path="$2"
      shift
      ;;
    --conversation_template)
      conversation_template="$2"
      shift
      ;;
    -o|--output_lora_path)
      output_dir="$2"
      shift
      ;;
    --deepspeed_args)
      deepspeed_args="$2"
      shift
      ;;
    --trust_remote_code)
      trust_remote_code="$2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done

# Finetune
exp_id=finetune_with_lora
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed --include localhost:2,3,4,5,6,7 ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code ${trust_remote_code} \
    --lora_model_path ${lora_model_path} \
    --dataset_path ${dataset_path} \
    --conversation_template ${conversation_template} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --block_size 512 \
    --per_device_train_batch_size 16 \
    --gradient_checkpointing 1 \
    --use_lora 1 \
    --lora_r 16 \
    --save_aggregated_lora 0\
    --deepspeed configs/ds_config_zero2.json \
    --fp16 \
    --run_name ${exp_id} \
    --validation_split_percentage 0 \
    --logging_steps 100 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 8 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
