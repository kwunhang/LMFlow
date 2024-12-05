#!/bin/bash

# Define the version
version=14b_FP_10e_v2

# Define paths using the version variable
output_dir=/ssddata/jimmy/teleQNA-arena/$version
teleQNA_output_path=/ssddata/jimmy/TeleQnA/${version}_answer.txt # Correctly uses the version for the file path

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DS_SKIP_CUDA_CHECK=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=docker0
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export WANDB_MODE=disabled

# RUNNING THE FIRST SCRIPT: 14b_LORA4_1e.sh

# Define and initialize default variables for finetuning
model_name_or_path=/ssddata/jimmy/model/Qwen2.5-14B-Instruct
dataset_path=/ssddata/cug-llm-data/sft_teleQNA
conversation_template=qwen
deepspeed_args="--master_port=11000"
trust_remote_code=0

# Parse arguments for finetuning
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

# Finetune model
exp_id=finetune_$version
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed  --include localhost:2,3 ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code ${trust_remote_code} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --conversation_template ${conversation_template} \
    --num_train_epochs 10 \
    --learning_rate 2e-5 \
    --disable_group_texts 1 \
    --gradient_checkpointing \
    --block_size 256 \
    --per_device_train_batch_size 16 \
    --deepspeed configs/ds_config_zero3.json \
    --fp16 \
    --run_name finetune \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 1 \
    > >(tee ${log_dir}/train.log) \
    2> >(tee ${log_dir}/train.err >&2)


# RUNNING THE SECOND SCRIPT: test_14b_LORA4_1e.sh

export CUDA_VISIBLE_DEVICES=2,3

sleep 1m

# Use the finetuning output directory as the model path for inference
model_name_or_path=$output_dir

# Define and initialize default variables for inference
run_name=vllm_inference
dataset_path=data/alpaca/test_conversation
output_dir=data/inference_results  # Different from the finetune output_dir
output_file_name=results.json
apply_chat_template=True
trust_remote_code=0

# Parse arguments for inference
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
  --use_beam_search False \
  --temperature 0.7 \
  --top_p 0.9 \
  --max_new_tokens 1024 \
  --enable_decode_inference_result True \
  --vllm_gpu_memory_utilization 0.95 \
  --vllm_tensor_parallel_size 2 \
  --teleQNA_output_path ${teleQNA_output_path} \
  2>&1 | tee ${log_dir}/vllm_inference.log \
  > /home/jimmy/LLM-FT/teleQNA_result/acc_${version}.txt

# Optionally, redirect results to a file
# ./jimmy/teleQNA/test_script/test_14b_LORA4_1e.sh > /home/jimmy/LLM-FT/teleQNA_result/acc_${version}.txt
