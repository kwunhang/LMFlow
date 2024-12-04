#!/bin/bash
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.

# Parses arguments
run_name=vllm_inference
model_name_or_path=/ssddata/jimmy/teleQNA-arena/14bm_LORA4_3e
# model_name_or_path='/ssddata/jimmy/sft-model_2/sft2-loraft-Qwen2.5-14B-Instruct'
# model_name_or_path='/ssddata/jimmy/model/Qwen2.5-14B'
# lora_model_path=/ssddata/jimmy/ft-model/sft-tele-Qwen2.5-14B
# lora_model_path='/ssddata/jimmy/ft-model/sft-tele-Qwen2.5-14B_lrank4'
dataset_path=data/alpaca/test_conversation
output_dir=data/inference_results
output_file_name=results.json
apply_chat_template=True

# Safety related arguments
trust_remote_code=0

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

# inference
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${run_name}
output_file_path=${output_dir}/${run_name}/${output_file_name}
mkdir -p ${output_dir}/${run_name} ${log_dir}

python jimmy/inference/teleqna_run.py \
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
  2>&1 | tee ${log_dir}/vllm_inference.log