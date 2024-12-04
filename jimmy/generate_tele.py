import os 
import json
import numpy as np
import pandas as pd
import random

# model = "gpt-3.5-turbo"
model = "DSF-CUG-LLM"
questions_path = "/ssddata/jimmy/data/TeleQnA.txt"
save_path = os.path.join("/ssddata/jimmy/TeleQnA/" + "teleQNA_explain_generation.txt")

# n_questions = 5 # Batch the questions asked to reduce time
max_attempts = 1 # Maximal number of trials before skipping the question

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

# answer generation function

from copy import deepcopy
import json
import openai
import ast
from openai import AsyncOpenAI, OpenAI, APIConnectionError, RateLimitError


LLM_BASE_URL = "http://localhost:8000/v1/"
LLM_API_KEY = "sk-22"
MODEL = "DSF-CUG-LLM"

openai.api_key = " " ## Insert OpenAI's API key
openai_async_client = OpenAI(
        api_key=LLM_API_KEY, base_url=LLM_BASE_URL
    )
syst_prompt_1 = """
Please provide the answers explanation and related knowledge to the following telecommunications related question-answer pair. The question, answer and short explanation are provided. Your answers must be in a string format with detail explanation:
"""

# syst_prompt_1 = """
# Please provide the answers explanation and related knowledge to the following telecommunications related question-answer pair. The question, answer and short explanation are provided. Your answers must be in a string format with detail explanation:
# """

# 1: explaination
# 2: with option, chain of though
# 3: formatting

def generate_answer_with_LLM(questions_dict, model):
    questions = deepcopy(questions_dict)
    answers_only = {}
    generate_questions_answer = {}
    for q in questions_dict:
        answers_only[q] = {
            "question": questions_dict[q]["question"],
            "answer": questions_dict[q]["answer"]
        }
    
        
        if 'category' in questions:
            questions[q].pop('category')

        user_prompt = f"Here are the questions: {questions[q]['question']} \nAnswer: {questions[q]['answer']} \nShort explanation: {questions[q]['explanation']} \n"
        
        generated_output = openai_async_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": syst_prompt_1},
                {"role": "user", "content": user_prompt},
            ]
        )
        
        predicted_answers_str = generated_output.choices[0].message.content

        predicted_answers_str = predicted_answers_str.replace('"\n', '",\n')

        generate_questions_answer[q] = {
            'question': questions_dict[q]["question"],
            'answer': predicted_answers_str
        }


    return generate_questions_answer



print("Start at question: {}".format(start))
print("End at question: {}".format(end))


k = 0

for start_id in range(start, end, 1):
    attempts = 0
    end_id = np.minimum(start_id + 1, len(all_questions)-1)
            
    q_names = ["question {}".format(shuffled_idx[k]) for k in range(start_id, end_id)]
    selected_questions = {}
    
    for q_name in q_names:
        selected_questions[q_name] = all_questions[q_name]

    while attempts < max_attempts:

        try:
            generate_questions_answer = generate_answer_with_LLM(selected_questions, model)
            
            for q in selected_questions:  
                generate_questions_answer[q]['answer']
                results[q] = deepcopy(selected_questions[q])
                results[q]['generate answer'] = generate_questions_answer[q]['answer']
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
        print("generated:",k)
        with open(save_path, 'w') as f:
            res_str = json.dumps(results)
            f.write(res_str)

with open(save_path, 'w') as f:
    res_str = json.dumps(results)
    f.write(res_str)


# print("Total number of questions answered: {}".format(len(categories)))
# print(summary)
print()
print("Finished generation")
# print("Final result: {}".format(np.mean([q['correct'] for q in results.values()])))