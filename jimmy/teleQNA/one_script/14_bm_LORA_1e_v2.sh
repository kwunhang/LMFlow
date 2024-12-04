#!/bin/bash

# Define the version
version=14bm_LORA4_1e_v2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DS_SKIP_CUDA_CHECK=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=docker0 # Change this if needed
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export WANDB_MODE=disabled

# RUNNING THE FIRST SCRIPT: 14b_LORA4_1e.sh

# Define and initialize default variables
model_name_or_path=/ssddata/jimmy/model/Qwen2.5-14B-Instruct
lora_model_path=/ssddata/jimmy/sft-model/sft-loraft-Qwen2.5-14B-Instruct
dataset_path=/ssddata/cug-llm-data/sft_teleQNA
conversation_template=qwen
output_dir=/ssddata/jimmy/teleQNA-arena/$version
deepspeed_args="--master_port=11000"
trust_remote_code=0

# Parse arguments
# while [[ $# -ge 1 ]]; do
#   key="$1"
#   case ${key} in
#     -m|--model_name_or_path)
#       model_name_or_path="$2"
#       shift
#       ;;
#     -d|--dataset_path)
#       dataset_path="$2"
#       shift
#       ;;
#     --conversation_template)
#       conversation_template="$2"
#       shift
#       ;;
#     -o|--output_lora_path)
#       output_dir="$2"
#       shift
#       ;;
#     --deepspeed_args)
#       deepspeed_args="$2"
#       shift
#       ;;
#     --trust_remote_code)
#       trust_remote_code="$2"
#       shift
#       ;;
#     *)
#       echo "error: unknown option \"${key}\"" 1>&2
#       exit 1
#   esac
#   shift
# done

# # Finetune model
# exp_id=finetune_with_lora
# project_dir=$(cd "$(dirname $0)"/..; pwd)
# log_dir=${project_dir}/log/${exp_id}
# mkdir -p ${output_dir} ${log_dir}

# deepspeed --include localhost:0,1,2,3,4,5,6,7 ${deepspeed_args} \
#   examples/finetune.py \
#     --model_name_or_path ${model_name_or_path} \
#     --lora_model_path ${lora_model_path} \
#     --trust_remote_code ${trust_remote_code} \
#     --dataset_path ${dataset_path} \
#     --conversation_template ${conversation_template} \
#     --output_dir ${output_dir} --overwrite_output_dir \
#     --num_train_epochs 1 \
#     --learning_rate 1e-4 \
#     --block_size 512 \
#     --per_device_train_batch_size 16 \
#     --gradient_checkpointing 1 \
#     --use_lora 1 \
#     --lora_r 4 \
#     --save_aggregated_lora 1\
#     --deepspeed configs/ds_config_zero2.json \
#     --fp16 \
#     --run_name ${exp_id} \
#     --validation_split_percentage 0 \
#     --logging_steps 100 \
#     --do_train \
#     --ddp_timeout 72000 \
#     --save_steps 5000 \
#     --dataloader_num_workers 4 \
#     | tee ${log_dir}/train.log \
#     2> ${log_dir}/train.err

# RUNNING THE SECOND SCRIPT: test_14b_LORA4_1e.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Define and initialize default variables
model_name_or_path=$output_dir
run_name=vllm_inference
# model_name_or_path=/ssddata/jimmy/teleQNA-arena/14b_LORA4_1e
dataset_path=data/alpaca/test_conversation
output_dir=data/inference_results
output_file_name=results.json
apply_chat_template=True
teleQNA_output_path=/ssddata/jimmy/TeleQnA/${version}_answer.txt
trust_remote_code=0

# Parse arguments
while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -r|--run_name)
      run_name="$2"
      shift
      ;;
    -m|--model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    -d|--dataset_path)
      dataset_path="$2"
      shift
      ;;
    --output_dir)
      output_dir="$2"
      shift
      ;;
    --output_file_name)
      output_file_name="$2"
      shift
      ;;
    --apply_chat_template)
      apply_chat_template="$2"
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

# Perform inference
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${run_name}
output_file_path=${output_dir}/${run_name}/${output_file_name}
mkdir -p ${output_dir}/${run_name} ${log_dir}

python jimmy/teleQNA/teleqna_run_vllm_test.py \
  --use_vllm True \
  --trust_remote_code ${trust_remote_code} \
  --model_name_or_path ${model_name_or_path} \
  --random_seed 42 \
  --apply_chat_template ${apply_chat_template} \
  --num_output_sequences 2 \
  --temperature 0.7 \
  --top_p 0.9 \
  --max_new_tokens 1024 \
  --enable_decode_inference_result True \
  --vllm_gpu_memory_utilization 0.95 \
  --vllm_tensor_parallel_size 2 \
  --teleQNA_output_path ${teleQNA_output_path} \
  2>&1 | tee ${log_dir}/vllm_inference.log 
  # > /home/jimmy/LLM-FT/teleQNA_result/acc_${version}.txt

# Save results
# ./jimmy/teleQNA/test_script/test_14b_LORA4_1e.sh > /home/jimmy/LLM-FT/teleQNA_result/acc_14b_LORA4_1e.txt
