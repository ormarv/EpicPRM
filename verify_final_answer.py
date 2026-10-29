import re
from hendryck_cleanup import *
import multiprocessing as mp
import tempfile
import os
import shutil
import base64
import zlib
import pickle
import json
import ast
from vllm import SamplingParams

def string_matching(answer:str, letter:str):
    print(f"Letter: {letter}")
    extracted_answer = None
    patterns = [
        r"\\boxed\{([A-Z])\}",
        r"(?:answer)\s*[:\s]*\(?([A-Z])\)?",
        r"\b([A-Z])\b(?:\s*)$"
    ]
    for i,p in enumerate(patterns):
        match = re.search(p, answer, re.IGNORECASE)
        if match:
            print(f"Pattern used: {p}, n°{i}")
            extracted_answer = match.group(1).upper()
            break
    if extracted_answer and extracted_answer==letter:
        return True
    print(f"Extracted answer: {extracted_answer}")
    return False

def answer_matching(answer:str, gold:str):
    patterns = [
        r"\\boxed\{([A-Z0-9]+)\}",
        r"(?:answer)\s*[:\s]*\(?([A-Z0-9]+)\)?",
        r"\b([A-Z0-9]+)\b(?:\s*)$"
    ]
    for i,p in enumerate(patterns):
        match = re.search(p, answer, re.IGNORECASE)
        if match:
            print(f"Pattern used: {p}, n°{i}")
            extracted_answer = match.group(1).upper()
            break
    if extracted_answer and extracted_answer==gold:
        return True
    print(f"Extracted answer: {extracted_answer}")
    return False


def clean_answer(answer:str)->str:
    answer = re.sub(f"<think>.*?</think>", "", answer, flags=re.DOTALL)
    boxed_match = re.findall(r"\\boxed{(.*?)\}", answer)
    if boxed_match:
        return boxed_match[-1].strip()
    return answer.strip()

def grade_math(answer:str, gold_standard:str):
    cleaned_answer = last_boxed_only_string(answer)
    rm_ans = remove_boxed(cleaned_answer)
    equiv = is_equiv(rm_ans, gold_standard)
    print(f"Answer: {rm_ans}")
    print("\n")
    #print(f"Result: {type(result)}")
    print(f"Gold: {gold_standard}")
    return equiv



def grade_deltabench_code(solution:str, lcot:str, llm):
    # extract code from lcot
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", lcot, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(r"```(?:[a-zA-Z#+-]*)\n(.*?)```", lcot, re.DOTALL)
        if not code_blocks:
            print(f"No recognizable code block in: {lcot}")
            return False
    extracted_code = "\n".join(code_blocks).strip()
    try:
        ast.parse(extracted_code)
        prompt = f"""You are an expert code evaluator. Compare the Reference Solution with the model's generated Long Chain-of-Thought (LCoT) and extracted code to determine if the final answer is functionally correct.

        Reference Solution:
        {solution}

        Model's Long Chain-of-Thought (LCoT):
        {lcot}

        Extracted Code:
        {extracted_code}

        Evaluate whether the model's solution correctly solves the problem with equivalent logic to the reference solution.
        You must respond strictly in the following format:
        VERDICT: YES
        REASON: [brief explanation]

        OR

        VERDICT: NO
        REASON: [brief explanation]
        """
        sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
        outputs = llm.generate([prompt], sampling_params)
        response_text = outputs[0].outputs[0].text.strip()
        if "verdict: yes" in response_text.lower():
            return True
        else:
            return False
    except:
        print(f"Failed AST syntax check.")
        return False

def judge_deltabench_reasoning(solution:str, last_reasoning_section:str, llm)->bool:
    prompt = f"""You are an expert evaluator. Compare the Reference Solution with the model's final answer (provided in the final section of its reasoning trace) to determine if the final conclusion is correct.

Reference Solution:
{solution}

Model's Final Answer / Last Section:
{last_reasoning_section}

Evaluate whether the model's final answer matches or is logically equivalent to the reference solution. Minor formatting or notation differences are acceptable, but the core conclusion/value must be correct.

You must respond strictly in the following format:
VERDICT: YES
REASON: [brief explanation]

OR

VERDICT: NO
REASON: [brief explanation]
"""
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    outputs = llm.generate([prompt], sampling_params)
    response_text = outputs[0].outputs[0].text.strip()
    if "verdict: yes" in response_text.lower():
        return True
    else:
        return False

def grade_answer(answer:str, gold_standard:str, dataset:str, llm)->bool:
    if dataset in ["gpqa", "mmlu"]:
        label = string_matching(answer, gold_standard)
    elif dataset == "math":
        label = grade_math(answer, gold_standard)
    elif dataset == "deltabench_mcq":
        label = answer_matching(answer, gold_standard)
    elif dataset == "deltabench_code":
        label = grade_deltabench_code(gold_standard, answer, llm)
    else:  # LLM-as-a-Judge
        label = judge_deltabench_reasoning(gold_standard, answer, llm)
    return label
