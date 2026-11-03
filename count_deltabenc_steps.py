#!/usr/bin/env python3
from tqdm import tqdm
from datasets import load_dataset
import re

deltabench_path = "../.cache/huggingface/hub/datasets--OpenStellarTeam--DeltaBench/snapshots/894d233e5beb06b312b29761cc10e10ca5d2588a/"

dataset = load_dataset(deltabench_path)["train"]

math_lcots = 0
code_lcots = 0
science_lcots = 0
gen_lcots = 0

math_steps = {"correct": 0, "incorrect":0}
code_steps = {"correct": 0, "incorrect": 0}
science_steps = {"correct": 0, "incorrect": 0}
gen_steps = {"correct": 0, "incorrect": 0}

for sample in tqdm(dataset):
    sections_list = re.split(r"section[0-9]: ",sample["sections_content"])
    full_reasoning = "".join(sections_list)
    steps_list = full_reasoning.split("\n\n")
    error_step_raw = re.compile(r"'error_step': [a-zA-Z]*([0-9]+)").search(sample["sections_labeled_info"])
    if error_step_raw:
        error_step = int(error_step_raw.group(1))-1
    else:
        print(f"Did not find an error step in {sample["sections_labeled_info"]}.")
        continue
    correct_steps_list = steps_list[:error_step]
    incorrect_steps_list = steps_list[error_step:]
    subject = sample["task_l1"]
    if subject == "math":
        math_lcots += 1
        math_steps["correct"] += len(correct_steps_list)
        math_steps["incorrect"] += len(incorrect_steps_list)
    elif subject == "code":
        code_lcots += 1
        code_steps["correct"] += len(correct_steps_list)
        code_steps["incorrect"] += len(incorrect_steps_list)
    elif subject == "general":
        gen_lcots += 1
        gen_steps["correct"] += len(correct_steps_list)
        gen_steps["incorrect"] += len(incorrect_steps_list)
    elif subject == "Physics&Chemistry&Biology":
        science_lcots += 1
        science_steps["correct"] += len(correct_steps_list)
        science_steps["incorrect"] += len(incorrect_steps_list)

print(f"Math: {math_lcots} LCoTs, {math_steps}")
print(f"Code: {code_lcots} LCoTs, {code_steps}")
print(f"Science: {science_lcots} LCoTs, {science_steps}")
print(f"General reasoning: {gen_lcots} LCoTs, {gen_steps}")