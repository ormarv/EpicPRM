import re
from hendryck_cleanup import *
from lcb_runner.evaluation.testing_util import run_test
from lcb_runner.utils.extraction_utils import extract_test_output_code
import multiprocessing as mp
import tempfile
import os
import shutil
import base64
import zlib
import pickle
import json

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

def worker(queue, samp, test_code, temp_dir):
    try:
        os.chdir(temp_dir)
        results, metadata = run_test(samp, test=test_code)
        queue.put((results, metadata))
    except Exception as e:
        queue.put(([False], str(e)))

def run_test_isolated(sample, code, timeout=6):
    context = mp.get_context("spawn")

    q = context.Queue()
    temp_dir = tempfile.mkdtemp()
    p = context.Process(target=worker, args=(q, sample, code, temp_dir))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        shutil.rmtree(temp_dir, ignore_errors=True)
        return [False], {"error":"Timeout or Infinite Loop"}
    
    if not q.empty():
        return q.get()
    else:
        return [False], {"error":"Process crashed silently"}

def grade_lcb(answer:str, sample):
    code = extract_test_output_code(answer)
    # Now we get the private test cases from the encrypted string
    inputs = []
    outputs = []
    decoded_bytes = base64.b64decode(sample["ptc"])
    decompressed_bytes = zlib.decompress(decoded_bytes)
    readable_cases = pickle.loads(decompressed_bytes)
    ptc = json.loads(readable_cases)
    for x in ptc:
        if len(x)>0:
            inputs.append(x["input"])
            outputs.append(x["output"])
        else:
            print(f"Empty x: {x}")
            continue
    s = {'input_output':json.dumps({'inputs':inputs, 'outputs':outputs, 'fn_name':sample["fn_name"]})}
    print(f"Code: {code}\n")
    if not code:
        print(f"No code found.")
        return False
    #str_sample = json.dumps(sample)
    results, metadata = run_test_isolated(s, code)
    print(f"Results: {results}, Metadata: {metadata}")
    # Let's assume that results is a list of bools
    if not results:
        return False
    for result in results:
        if result is not True:
            return False
    return True

def grade_answer(answer:str, gold_standard:str|dict|None, letter:str|None, dataset:str)->bool:
    if dataset in ["gpqa", "mmlu"]:
        label = string_matching(answer, letter)
    elif dataset == "math":
        label = grade_math(answer, gold_standard)
    else:  # LCB
        label = grade_lcb(answer, gold_standard)  # The private test cases are in the gold standard.
    return label