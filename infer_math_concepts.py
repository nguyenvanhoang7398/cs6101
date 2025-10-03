from accelerate import Accelerator
import json, math, time
import torch
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM


def read_jsonl_file(file_path):
    """
    Reads a file where each line is a JSON object (often called JSON Lines or JSONL)
    and returns a list of dictionaries.
    """
    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Strip leading/trailing whitespace and check if the line is not empty
                clean_line = line.strip()
                if clean_line:
                    try:
                        # Use json.loads to parse the JSON string into a Python object (dictionary)
                        data_object = json.loads(clean_line)
                        data_list.append(data_object)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON on line: {clean_line[:50]}...")
                        print(f"Error details: {e}")
                        # Optionally, you can choose to skip or raise an error here
                        continue
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    return data_list

def infer_prm800k():
    prm_path = "dataset/prm800k_phase2_test.jsonl"
    prm_data = read_jsonl_file(prm_path)[:10]
    all_completions = []
    for question in prm_data:
        if "label" in question and "steps" in question["label"]:
            for step in question["label"]["steps"]:
                if "completions" in step:
                    for completion in step["completions"]:
                        if "text" in completion:
                            all_completions.append(completion)
    
    prompt = "List all mathematical concepts that you need to understand in this problems, each concept is not more than 5 words and write [None] if there is not any. Problem: {text}."

    completion_prompts = [
        prompt.format(text=x["text"])
    for x in all_completions][:2]
    print(completion_prompts)

    math_concepts = infer_llama_awq(completion_prompts)
    print(math_concepts[:2])


def infer_llama_awq(texts):
    print("Text size: {}".format(len(texts)))
    
    model_name_or_path = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    acc = Accelerator()
    local_rank = acc.local_process_index
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoAWQForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    world_size = acc.num_processes
    shard_size = math.ceil(len(texts)/world_size)
    start = local_rank * shard_size
    end = min(start+shard_size, len(texts))
    print("Dataset size: {} local process index: {} world size: {} shard size: {} start: {} end: {}".format(
        len(texts), local_rank, world_size, shard_size, start, end
    ))
    proc_texts = texts[start:end]
    inference_outputs = []
    batch_idx, batch_size = 0, 16

    while batch_idx < shard_size:
        batch_texts = proc_texts[batch_idx:batch_idx+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(acc.device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, do_sample=True, max_new_tokens=128)
            decoded_outputs = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            inference_outputs.extend(decoded_outputs)
        batch_idx += 1
    
    gathered = acc.gather_for_metrics(inference_outputs)
    main_outputs = []
    if acc.is_main_process:
        main_outputs = [t for t in gathered if t is not None]
    return main_outputs


if __name__ == "__main__":
    infer_prm800k()
