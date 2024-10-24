#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A simple shell chatbot implemented with lmflow APIs.
"""
import logging
import json
import os
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
import warnings

from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments
import pandas as pd

logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")


file_path = "/ssddata/jimmy/data/运维知识.csv"
# output_file = "/ssddata/jimmy/result/sft/lora-sft-Qwen1.5-14B-Chat-运维知识-answer.csv"
# output_file = "/ssddata/jimmy/result/sft_2/lora-sft2-Qwen1.5-14B-Chat-运维知识-answer.csv"
# output_file = "/ssddata/jimmy/result/sft_2_2/lora-sft2_2-Qwen1.5-14B-Chat-运维知识-answer.csv"
output_file = "/ssddata/jimmy/result/sft_2_3/lora-sft2_3-Qwen2.5-14B-Chat-运维知识-answer.csv"

def main():    
    pipeline_name = "inferencer"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((
        ModelArguments,
        PipelineArguments,
    ))
    model_args, pipeline_args = parser.parse_args_into_dataclasses()
    inferencer_args = pipeline_args

    with open (pipeline_args.deepspeed, "r") as f:
        ds_config = json.load(f)

    model = AutoModel.get_model(
        model_args,
        tune_strategy='none',
        ds_config=ds_config,
        device=pipeline_args.device,
        use_accelerator=True,
    )

    # We don't need input data, we will read interactively from stdin
    data_args = DatasetArguments(dataset_path=None)
    dataset = Dataset(data_args)

    inferencer = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )

    df = pd.read_csv(file_path)

    # Inferences
    model_name = model_args.model_name_or_path
    if model_args.lora_model_path is not None:
        model_name += f" + {model_args.lora_model_path}"
    results = []
    
    # system_message = "你是一个人工智能助手，用户会就不同类型的问题向你询问，请根据问题的类型选择相应的结构来回答。例如：告警解释, 现象描述, 日志含义, 错误码描述, 对系统的影响, 可能原因, 处理步骤, 处理方法。请根据问题的性质，选择3到4个相关部分来提供清晰、准确的回答。"

    # system_message = "你是一个人工智能助手，用户会就不同类型的问题向你询问，请根据问题的类型选择相应的结构来回答。例如：告警解释, 现象描述, 日志含义, 错误码描述, 对系统的影响, 可能原因, 处理步骤, 处理方法。请根据问题的性质，选择3到4个相关部分，以markdown的形式提供清晰、准确的回答。"
    system_message = "你是一个人工智能助手，用户会就不同类型的问题向你询问，请根据问题的类型选择相应的结构来回答。例如：告警解释, 现象描述, 日志含义, 错误码描述, 对系统的影响, 可能原因, 处理步骤, 处理方法。请根据问题的性质，选择3到4个相关部分以1，2，3，4分点回答,如果需要继续分点请以a: ,b: ,c: ,d: 开始。"

    for _, cur_row in df.iterrows():
        input_text = f"INPUT:{cur_row['Question']}"
        print(input_text)

        input_dataset = dataset.from_dict({
            "type": "text_only",
            # "instances": [ { "text": input_text } ]
            "instances": [ { "text": f"""<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n""" } ]
        })

        output_dataset = inferencer.inference(
            model=model,
            dataset=input_dataset,
            max_new_tokens=inferencer_args.max_new_tokens,
            temperature=inferencer_args.temperature,
        )
        output = output_dataset.to_dict()["instances"][0]["text"]
        if (output == None or output == ""):
            print("empty output")
            print("imput dataset:")
            print(input_dataset)
        results.append({"Question": input_text, "LLM Answer": output})
    results_df = pd.DataFrame(results)
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results_df.to_csv(output_file, index=False)
        # print(output)


if __name__ == "__main__":
    main()
