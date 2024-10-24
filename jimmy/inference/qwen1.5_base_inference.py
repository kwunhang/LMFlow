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
output_file = "/ssddata/jimmy/result/base/Qwen1.5-14B-Chat-运维知识-answer.csv"
    
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
    
    system_message = "你是一個人工智能助手，用戶會就错误信息向你詢問問題，請按以下格式輸出\n## 对系统的影响\n## 可能原因\n## 处理步骤\n以下是例子：\問題：ALM-1821 IUA链路集故障問題是什麼？\n回答：## 对系统的影响\n该链路集无法承载业务消息。业务消息无法通过该链路集发送。\n## 可能原因\nIUA链路集中所有链路被误删除或IUA链路集中所有链路故障。\n## 处理步骤\n使用LST MIUALKS命令检查链路集中是否配置了IUA链路。是 =>2否 =>4使用DSP MIUALNK命令检查同一链路集下各链路的状态。所有链路的状态都不是为“已激活” =>3至少有一条链路的状态为“已激活” =>7使用ACT MIUALNK命令激活至少一条IUA链路。查看告警是否恢复。是 => 结束否 =>5使用ACT MIUALNK命令增加IUA链路，检查告警是否恢复。是 => 结束否 =>7检查是否存在与该链路集相关的“ALM-1819 IUA链路故障”告警。是 =>6否 =>7参见“ALM-1819 IUA链路故障”的处理步骤进行处理，恢复故障的链路。检查此告警是否恢复。是 => 结束否 =>7请联系华为技术支持处理。"

    for _, cur_row in df.iterrows():
        input_text = cur_row['Question']

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
