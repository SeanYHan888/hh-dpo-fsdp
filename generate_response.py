import torch
from transformers import AutoTokenizer
import numpy as np
import random
from datasets import load_dataset
from tqdm import tqdm
import os
import yaml
import json
from process_hh_dataset import build_HH_dataset
import argparse
from vllm import LLM, SamplingParams


# load yaml config
def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# control randomness
def random_controler(seed):
    np.random.seed(seed)
    random.seed(seed)


# vllm sampling params
def get_vllm_sampling_params(
    do_sample, temperature, top_p, max_prompt_length, max_new_tokens
):
    # greedy decoding
    if not do_sample:
        temperature = 0.0
        top_p = 1.0

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        truncate_prompt_tokens=int(max_prompt_length),
    )
    return sampling_params


# build vllm model
def build_vllm_model(model_path, use_bf16, tensor_parallel_size):
    dtype = "bfloat16" if use_bf16 else "float16"

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        trust_remote_code=True,
    )
    return llm


# define chunks
def chunked(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield i, lst[i : i + chunk_size]


# build write jsonl
def build_write_jsonl(out_path, records):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# generate
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_yaml_config(args.config)

    use_bf16 = config["precision"] == "bf16"

    # vllm parallism distribution
    tensor_parallel_size = config["vllm"]["tensor_parallel_size"]
    chunk_size = config["vllm"]["chunk_size"]

    # load tokenizer
    base_model_name = config["ref_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # test config
    test_cfg = config["HH_test"]
    seed = test_cfg["seed"]
    random_controler(seed)

    # load test data
    num_sample = test_cfg["num_sample"]
    ds_name = test_cfg["hh_test"]
    raw_hh = load_dataset(ds_name, data_dir="harmless-base", split="test")
    hh_test = build_HH_dataset(raw_hh)
    hh = hh_test.shuffle(seed=seed).select(range(num_sample))

    # generation setup
    max_prompt_length = int(test_cfg["max_prompt_length"])
    max_new_tokens = int(test_cfg["max_new_tokens"])
    temperature = float(test_cfg["temperature"])
    top_p = float(test_cfg["top_p"])
    do_sample = test_cfg["do_sample"]

    # llm params
    sampling_params = get_vllm_sampling_params(
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
    )

    prompts = [ex["prompt"] for ex in hh]
    chosens = [ex["chosen"] for ex in hh]

    # policy model generation
    policy_records = []
    policy_model_path = config["dpo_training"]["save_dir"]
    policy_llm = build_vllm_model(policy_model_path, use_bf16, tensor_parallel_size)

    total_chunks = (len(prompts) + chunk_size - 1) // chunk_size

    for start_idx, prompt_chunk in tqdm(
        chunked(prompts, chunk_size=chunk_size),
        desc="vLLM generating",
        total=total_chunks,
    ):
        policy_outs = policy_llm.generate(prompt_chunk, sampling_params)

        for i, out in enumerate(policy_outs):
            global_i = start_idx + i
            text = out.outputs[0].text.lstrip()

            policy_records.append(
                {
                    "idx": global_i,
                    "prompt": prompts[global_i],
                    "response": text,
                    "HH_response": chosens[global_i],
                    "model_id": "policy_model",
                    "dataset": "Anthropic/hh-rlhf",
                    "split": "test",
                    "max_prompt_length": max_prompt_length,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": bool(do_sample),
                    "temperature": float(temperature) if do_sample else 0.0,
                    "top_p": float(top_p) if do_sample else 1.0,
                }
            )

    policy_out_dir = test_cfg["HH_test_policy_out"]
    build_write_jsonl(policy_out_dir, policy_records)

    # base model generation
    del policy_llm

    base_records = []
    base_llm = build_vllm_model(
        base_model_name, use_bf16=use_bf16, tensor_parallel_size=tensor_parallel_size
    )

    for start_idx, prompt_chunk in tqdm(
        chunked(prompts, chunk_size=chunk_size),
        desc="vLLM generating (base)",
        total=total_chunks,
    ):
        base_outs = base_llm.generate(prompt_chunk, sampling_params)

        for i, out in enumerate(base_outs):
            global_i = start_idx + i
            text = out.outputs[0].text.lstrip()

            base_records.append(
                {
                    "idx": global_i,
                    "prompt": prompts[global_i],
                    "response": text,
                    "HH_response": chosens[global_i],
                    "model_id": "base_model",
                    "dataset": "Anthropic/hh-rlhf",
                    "split": "test",
                    "max_prompt_length": max_prompt_length,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": bool(do_sample),
                    "temperature": float(temperature) if do_sample else 0.0,
                    "top_p": float(top_p) if do_sample else 1.0,
                }
            )

    base_out_dir = test_cfg["HH_test_base_out"]
    build_write_jsonl(base_out_dir, base_records)

    print("vLLM generation completed.")


if __name__ == "__main__":
    main()
