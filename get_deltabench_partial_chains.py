#!/usr/bin/env python3
from vllm import LLM
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import json
import re
from generate_train_data import generate_prm_data
from utils import ag_step
import numpy as np
import itertools
from typing import List, Dict, Any, cast
from tqdm import tqdm
import matplotlib.pyplot as plt
from random import randint

def print_samples(filepath:str, samples:List[Dict[str,Any]])->None:
    with open(filepath,"w+") as f:
        for sample in samples:
            f.write(json.dumps(sample)+'\n')

class MyArgs():

    def __init__(self, dict_args):
        self.data_dir = dict_args["data_dir"]
        self.save_dir = dict_args["save_dir"]
        self.model_dir = dict_args["model_dir"]
        self.seed = dict_args["seed"]
        self.start = dict_args["start"]
        self.end = dict_args["end"]
        self.synthetic_data_dir = dict_args["synthetic_data_dir"]
        self.low_level = dict_args["low_level"]
        self.use_adaptive_binary = dict_args["use_adaptive_binary"]
        self.sequential_search = dict_args["sequential_search"]
        self.nb_keywords = dict_args["nb_keywords"]

def plot_stacked_bar_chart(scores, labels, categories,subject):
    scores_array = np.array(scores)
    labels_array = np.array(labels)

    counts_0, counts_1 = [], []

    for cat in categories:
        mask = np.isclose(scores_array, cat)
        counts_0.append(np.sum(mask & (labels_array == 0)))
        counts_1.append(np.sum(mask & (labels_array == 1)))

    x = np.arange(len(categories))

    plt.figure(figsize=(10,10))
    plt.bar(x, counts_0, label="Incorrect", color='red')
    plt.bar(x, counts_1, bottom=counts_0, label = "Correct", color = "green")
    plt.xlabel("Threshold")
    plt.ylabel("Number of steps")
    plt.xticks(x, [str(c) for c in categories])
    plt.savefig(f"contribution_threshold_study_deltabench_{subject}.png")

deltabench_path = "../.cache/huggingface/hub/datasets--OpenStellarTeam--DeltaBench/snapshots/894d233e5beb06b312b29761cc10e10ca5d2588a/"

dataset = load_dataset(deltabench_path)["train"]

scored_math_samples = []
scored_code_samples = []
scored_physchem_samples = []
scored_gen_samples = []


generate_prm_data_object = generate_prm_data(MyArgs({
    "data_dir":"",
    "save_dir":"",
    "model_dir":"",
    "seed": 42,
    "start":0,
    "end":10000,
    "synthetic_data_dir":"",
    "low_level": False,
    "use_adaptive_binary": True,
    "sequential_search": False,
    "nb_keywords": 8}))

gpu_memory_utilization_list = [0.45,0.45]

llm_dir_list=["../.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/","../.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/"]

llm_list = [LLM(
            model = llm_dir_list[i],
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization_list[i],
            max_model_len=16384,
            swap_space=16,
            seed=69) for i in range(len(llm_dir_list))]

math_scores = []
code_scores = []
physchem_scores = []
gen_scores = []

math_labels = []
code_labels = []
physchem_labels = []
gen_labels = []

tokenizer_list = [AutoTokenizer.from_pretrained(model) for model in llm_dir_list]
with open("clean_deltabench_samples.jsonl", "w+") as f:
    for item in tqdm(dataset):
        sample = cast(Dict[str, Any], item)
        print(f"sample: {sample}")
        subject = sample["task_l1"]
        if subject == "math" and len(scored_math_samples) >= 400:
            continue
        elif subject == "code" and len(scored_code_samples) >= 400:
            continue
        elif subject == "general" and len(scored_gen_samples) >= 400:
            continue
        elif subject == "Physics&Chemistry&Biology" and len(scored_physchem_samples) >= 400:
            continue
        gold_answer = sample["answer"] if sample["answer"] else sample["solution"]
        clean_sample = {"id":sample["id"], "subject": subject, "question": sample["question"], "gold_answer": gold_answer}
        sections_list = re.split(r"section[0-9]: ",sample["sections_content"])
        full_reasoning = "".join(sections_list)
        steps_list = full_reasoning.split("\n\n")
        error_step_raw = re.compile(r"'error_step': [a-zA-Z]*([0-9]+)").search(sample["sections_labeled_info"])
        if error_step_raw:
            error_step = int(error_step_raw.group(1))-1
        else:
            print(f"Did not find an error step in {sample["sections_labeled_info"]}.")
            continue

        clean_sample["steps"] = steps_list
        clean_sample["error_step"] = error_step
        if subject == "math":
            ds = "math"
        elif "gpqa" in sample["origin"].lower() or "agieval" in sample["origin"].lower():
            ds = "gpqa"
            gold_answer = gold_answer.split("\\boxed{")[1].split("}")[0] if "\\boxed{" in gold_answer else gold_answer
        else:
            ds = "deltabench"

        sample_num_n = 2
        # When creating the ag_step for the question, only the question is given, no current step.
        first_step = ag_step(question=clean_sample["question"], ans=gold_answer, dataset=ds, previous_steps="")

        # Now we sample the index of our incorrect step, create the corresponding node, and get its perplexity score
        if error_step < len(steps_list)-1:
            incorrect_step_index = randint(error_step, len(steps_list)-1)
        elif error_step == len(steps_list)-1:
            incorrect_step_index = error_step
        else:
            raise Exception(f"The error_step {error_step} is higher than the last step {len(steps_list)-1}.")
        incorrect_step = ag_step(question=clean_sample["question"], ans=gold_answer, dataset=ds, previous_steps="\n\n".join(steps_list[:incorrect_step_index]), cur_step = steps_list[incorrect_step_index])
        _, incorrect_step_perplexity_list, _, _ = generate_prm_data_object.verify_step(incorrect_step, llm_list, tokenizer_list, sample_num_n, "", "")
        incorrect_step_mc_ppl = sum(incorrect_step_perplexity_list)

        # and the index of our correct step
        correct_step_index = randint(0, error_step)
        correct_step = ag_step(question=clean_sample["question"], ans=gold_answer, dataset=ds, previous_steps="\n\n".join(steps_list[:correct_step_index]), cur_step = steps_list[correct_step_index])
        _, correct_step_perplexity_list, _, _ = generate_prm_data_object.verify_step(correct_step, llm_list, tokenizer_list, sample_num_n, "", "")
        correct_step_mc_ppl = sum(correct_step_perplexity_list) 

        # We get the perplexity score of the question
        _, perplexity_list, _, _ = generate_prm_data_object.verify_step(first_step, llm_list, tokenizer_list, sample_num_n, "", "")

        question_mc_ppl = sum(perplexity_list)

        # We get the contribution of each step

        if question_mc_ppl > 0:
            contribution_incorrect_step = incorrect_step_mc_ppl/question_mc_ppl
            contribution_correct_step = correct_step_mc_ppl/question_mc_ppl
        else:  #Unsure what else to do here.
            contribution_incorrect_step = 0
            contribution_correct_step = 0



        incorrect_sample = {"id":sample["id"], "subject": subject, "question": sample["question"], "gold_answer": sample["answer"], "previous_steps":"\n\n".join(steps_list[:incorrect_step_index]), "cur_step": steps_list[incorrect_step_index], "llm_dir_list": llm_dir_list, "step_index":incorrect_step_index, "contribution": contribution_incorrect_step}
        correct_sample = {"id":sample["id"], "subject": subject, "question": sample["question"], "gold_answer": sample["answer"], "previous_steps":"\n\n".join(steps_list[:correct_step_index]), "cur_step": steps_list[correct_step_index], "llm_dir_list": llm_dir_list, "step_index":correct_step_index, "contribution": contribution_correct_step}

        if subject == "math":
            scored_math_samples.append(incorrect_sample)
            math_scores.append(contribution_incorrect_step)
            scored_math_samples.append(correct_sample)
            math_scores.append(contribution_correct_step)
            math_labels.append(0)
            math_labels.append(1)
        elif subject == "code":
            scored_code_samples.append(incorrect_sample)
            code_scores.append(contribution_incorrect_step)
            scored_code_samples.append(correct_sample)
            code_scores.append(contribution_correct_step)
            code_labels.append(0)
            code_labels.append(1)
        elif subject == "general":
            scored_gen_samples.append(incorrect_sample)
            gen_scores.append(contribution_incorrect_step)
            scored_gen_samples.append(correct_sample)
            gen_scores.append(contribution_correct_step)
            gen_labels.append(0)
            gen_labels.append(1)
        else:
            scored_physchem_samples.append(incorrect_sample)
            physchem_scores.append(contribution_incorrect_step)
            scored_physchem_samples.append(correct_sample)
            physchem_scores.append(contribution_correct_step)
            physchem_labels.append(0)
            physchem_labels.append(1)

        #f.write(json.dumps(clean_sample)+ '\n')

print_samples("deltabench_math_samples_with_contribution.jsonl", scored_math_samples)
print_samples("deltabench_code_samples_with_contribution.jsonl", scored_code_samples)
print_samples("deltabench_gen_samples_with_contribution.jsonl", scored_gen_samples)
print_samples("deltabench_physchem_samples_with_contribution.jsonl", scored_physchem_samples)

# Now we print the number of correct and incorrect labels for each value of contribution

# First we round the scores to 1 decimal
math_scores = [round(s, 1) for s in math_scores]
code_scores = [round(s, 1) for s in code_scores]
physchem_scores = [round(s, 1) for s in physchem_scores]
gen_scores = [round(s, 1) for s in gen_scores]
categories = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.8, 0.9, 1.0]

plot_stacked_bar_chart(math_scores, math_labels, categories, "math")
plot_stacked_bar_chart(code_scores, code_labels, categories, "code")
plot_stacked_bar_chart(gen_scores, gen_labels, categories, "general")
plot_stacked_bar_chart(physchem_scores, physchem_labels, categories, "physchembio")
