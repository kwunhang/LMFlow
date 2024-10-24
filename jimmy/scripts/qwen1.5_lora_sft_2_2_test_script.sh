#!/bin/bash
# A simple chatbot script, the memory of the chatbot has a length of maximum
# model length, e.g. 4k for llama-2.

# model=/ssddata/jimmy/model/Qwen2.5-14B-Instruct
# adject the prompt
model=/ssddata/jimmy/sft-model_2/sft2-loraft-Qwen1.5-14B-Chat
lora_args=""
if [ $# -ge 1 ]; then
  model=$1
fi
if [ $# -ge 2 ]; then
  lora_args="--lora_model_path $2"
fi

    # --temperature 0.7 \
accelerate launch --config_file jimmy/configs/multigpu_config.yaml \
  jimmy/inference/qwen1.5_lora_sft_inference.py \
    --deepspeed configs/ds_config_chatbot.json \
    --model_name_or_path ${model} \
    --use_accelerator False \
    --max_new_tokens 2048 \
    --temperature 1.0 \
    # --end_string "#" \
    ${lora_args}
