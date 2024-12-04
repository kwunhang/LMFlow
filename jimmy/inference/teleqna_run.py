#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import logging
import os
import sys

from transformers import (
    HfArgumentParser
)

from lmflow.datasets import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.args import (
    ModelArguments, 
    DatasetArguments,
    AutoArguments,
)
questions_path = "/ssddata/jimmy/data/TeleQnA.txt"
# save_path = os.path.join("/ssddata/jimmy/TeleQnA/" + "sft_general_Qwen2_5_base_lmflow" + "_answers_all.txt")
# save_path = os.path.join("/ssddata/jimmy/TeleQnA/" + "sft_general_Qwen2_5_base_rank4" + "_answers_all.txt")
save_path = os.path.join("/ssddata/jimmy/TeleQnA/" + "sft_test" + "_answers.txt")

logger = logging.getLogger(__name__)
pipeline_name = "vllm_inferencer"
PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)
syst_prompt = """
Please provide the answers to the following telecommunications related multiple choice questions. The questions will be in a JSON format, the answers must also be in a JSON format as follows:
{
"question 1": {
"question": question,
"answer": "option {answer id}: {answer string}"
},
...
}
"""
def main():
    parser = HfArgumentParser((
        ModelArguments, 
        PipelineArguments
    ))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, pipeline_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, pipeline_args = parser.parse_args_into_dataclasses()

    data_args = DatasetArguments(dataset_path=None)
    dataset = Dataset(data_args)
    model = AutoModel.get_model(model_args, tune_strategy='none')

    def question_output(system_message, input_text):
        input_dataset = dataset.from_dict({
                "type": "text_only",
                # "instances": [ { "text": input_text } ]
                "instances": [ { "text": f"""<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n""" } ]
        })

        inferencer = AutoPipeline.get_pipeline(
                pipeline_name=pipeline_name,
                model_args=model_args,
                data_args=data_args,
                pipeline_args=pipeline_args
            )



        res = inferencer.inference(
            model=model,
            dataset=input_dataset,
            release_gpu=False,
            enable_decode_inference_result=pipeline_args.enable_decode_inference_result,
            enable_distributed_vllm_inference=False,
        )

        # print(res)

        output = res[0]['output'][0]
        return output

    # end of vllm





    #######

    # evaluations tool
    # teleQNA
    from copy import deepcopy
    import json
    # import openai
    import ast


    # openai.api_key = " " ## Insert OpenAI's API key
        
    syst_prompt = """
    Please provide the answers to the following telecommunications related multiple choice questions. The questions will be in a JSON format, the answers must also be in a JSON format as follows:
    {
    "question 1": {
    "question": question,
    "answer": "option {answer id}: {answer string}"
    },
    ...
    }
    """

    def check_questions_with_val_output(questions_dict, model):
        questions_only = deepcopy(questions_dict)
        answers_only = {}
        for q in questions_dict:
            answers_only[q] = {
                "question": questions_dict[q]["question"],
                "answer": questions_dict[q]["answer"]
            }
        
            questions_only[q].pop("answer")
            
            if 'explanation' in questions_only[q]:
                questions_only[q].pop('explanation')

            if 'category' in questions_only:
                questions_only[q].pop('category')
        
        user_prompt = "Here are the questions: \n "
        user_prompt += json.dumps(questions_only)
        
        generated_output = question_output(syst_prompt, user_prompt)
        
        predicted_answers_str = generated_output

        
        predicted_answers_str = predicted_answers_str.replace('"\n', '",\n')
        predicted_answers_str = predicted_answers_str[predicted_answers_str.find("{"):]
        
        parsed_predicted_answers = ast.literal_eval(predicted_answers_str)
        
        for q in parsed_predicted_answers:
            if "answer" in parsed_predicted_answers[q] and "question" in parsed_predicted_answers[q]:
                parsed_predicted_answers[q] = {
                    "question": parsed_predicted_answers[q]["question"],
                    "answer": parsed_predicted_answers[q]["answer"]
                }
        
        accepted_questions = {}
        
        for q in questions_dict:
            print(q)
            if q in parsed_predicted_answers and q in answers_only:
                print(q)
                if parsed_predicted_answers[q] == answers_only[q] and parsed_predicted_answers[q] != answers_only[q]:
                    print(f"\nInconsistency found in {q}:")
                    print(f"Question: {answers_only['question']}")
                    print(f"Tested Answer: {parsed_predicted_answers['question']}")
                    print(f"Answer: {answers_only['answer']}")
                    print(f"Tested Answer: {parsed_predicted_answers['answer']}")
                    print("-" * 80)
                if parsed_predicted_answers[q] == answers_only[q]:
                    accepted_questions[q] = questions_dict[q]
                else:
                    print(parsed_predicted_answers[q])
                    print(answers_only[q])

        return accepted_questions, parsed_predicted_answers



    ###########
    # teleQNA
    import os 
    import json
    import numpy as np
    import pandas as pd
    import random


    # model = "gpt-3.5-turbo"
    global questions_path
    save_path 

    n_questions = 5 # Batch the questions asked to reduce time
    max_attempts = 3 # Maximal number of trials before skipping the question

    print("Evaluating {}".format(model))

    with open(questions_path, encoding="utf-8") as f:
        loaded_json = f.read()
    all_questions = json.loads(loaded_json)

    end = len(all_questions)
    # end = 1000
    random.seed(1028)
    np.random.seed(seed=1028)
    shuffled_idx = np.arange(len(all_questions))

    if os.path.exists(save_path):
        with open(save_path) as f:
            loaded_json = f.read()
        results = json.loads(loaded_json)
        
        start = len(results)
        categories = [ques['category'] for ques in results.values()]
        correct = [ques['correct'] for ques in results.values()]
    else:
        results = {}
        start = 0
        categories = []
        correct = []
        

    print("Start at question: {}".format(start))

    k = 0

    for start_id in range(start, end, n_questions):
        attempts = 0
        end_id = np.minimum(start_id + n_questions, len(all_questions)-1)
                
        q_names = ["question {}".format(shuffled_idx[k]) for k in range(start_id, end_id)]
        selected_questions = {}
        
        for q_name in q_names:
            selected_questions[q_name] = all_questions[q_name]

        while attempts < max_attempts:
            try:
                accepted_questions, parsed_predicted_answers = check_questions_with_val_output(selected_questions, model)
                
                for q in selected_questions:  
                    parsed_predicted_answers[q]['answer']
                    results[q] = deepcopy(selected_questions[q])
                    results[q]['tested answer'] = parsed_predicted_answers[q]['answer']
                    results[q]['correct'] = q in accepted_questions
                    correct += [results[q]['correct']]
                    categories += [selected_questions[q]['category']]
            
                break
                
            except Exception as e:
                attempts += 1
                print(f"Attempt {attempts} failed. Error: {e}")
                print("Retrying...")
            
        else:
            print(f"Failed after {max_attempts} attempts.")

        k += 1
        if k % 5 == 0:
            with open(save_path, 'w') as f:
                res_str = json.dumps(results)
                f.write(res_str)

            res = pd.DataFrame.from_dict({
                'categories': categories,
                'correct': correct
            })

            summary = res.groupby('categories').mean()
            summary['counts'] = res.groupby('categories').count()['correct'].values
            
            print("Total number of questions answered: {}".format(len(categories)))
            print(summary)

    with open(save_path, 'w') as f:
        res_str = json.dumps(results)
        f.write(res_str)

    res = pd.DataFrame.from_dict({
        'categories': categories,
        'correct': correct
    })

    summary = res.groupby('categories').mean()
    summary['counts'] = res.groupby('categories').count()['correct'].values


    print("Total number of questions answered: {}".format(len(categories)))
    print(summary)
    print()
    print()
    print("Final result: {}".format(np.mean([q['correct'] for q in results.values()])))



if __name__ == "__main__":
    main()