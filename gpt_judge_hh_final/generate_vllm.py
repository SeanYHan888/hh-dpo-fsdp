"""
Generate model outputs on HH dataset using vLLM for fast batch inference.

Produces JSON compatible with judge_gpt4o.py for pairwise evaluation.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List

import numpy as np
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils import (
    build_hh_dataset,
    extract_instruction_from_prompt,
    format_prompt,
    load_hh_single_turn_instructions,
    parse_hh_to_messages,
)


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def get_sampling_params(
    do_sample: bool,
    temperature: float,
    top_p: float,
    max_prompt_length: int,
    max_new_tokens: int,
) -> SamplingParams:
    if not do_sample:
        temperature = 0.0
        top_p = 1.0
    return SamplingParams(
        temperature=temperature if temperature > 0 else 0.0,
        top_p=top_p if temperature > 0 else 1.0,
        max_tokens=max_new_tokens,
        truncate_prompt_tokens=max_prompt_length,
    )


def chunked(lst: List, chunk_size: int):
    for i in range(0, len(lst), chunk_size):
        yield i, lst[i : i + chunk_size]


def build_vllm_model(
    model_path: str,
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    tokenizer: str = None,
) -> LLM:
    llm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": dtype,
        "trust_remote_code": True,
    }
    if tokenizer:
        llm_kwargs["tokenizer"] = tokenizer
    return LLM(**llm_kwargs)


def main():
    parser = argparse.ArgumentParser(description="Generate model outputs using vLLM")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--model_name", type=str, default=None, help="Override model name"
    )
    parser.add_argument(
        "--output_file", type=str, default=None, help="Override output path"
    )
    parser.add_argument(
        "--max_instances", type=int, default=None, help="Override max examples"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    gen_cfg = config.get("generation", {})
    vllm_cfg = config.get("vllm", {})

    # Model settings
    model_name = args.model_name or gen_cfg.get("model_name")
    output_file = args.output_file or gen_cfg.get("output_file")
    if not model_name or not output_file:
        raise ValueError("model_name and output_file must be set")

    tokenizer_name = gen_cfg.get("tokenizer_name") or model_name
    tokenizer_fallback = gen_cfg.get("ref_model") or config.get("ref_model")

    # vLLM settings
    tensor_parallel_size = int(vllm_cfg.get("tensor_parallel_size", 1))
    chunk_size = int(vllm_cfg.get("chunk_size", 32))
    dtype = str(vllm_cfg.get("dtype", "bfloat16"))

    # Generation settings
    seed = gen_cfg.get("seed", 42)
    if seed:
        set_seed(int(seed))

    dataset_repo = gen_cfg.get("dataset_repo", "Anthropic/hh-rlhf")
    data_dir = gen_cfg.get("dataset_data_dir", "harmless-base")
    dataset_split = gen_cfg.get("dataset_split", "test")
    max_instances = args.max_instances or gen_cfg.get("max_instances")
    max_instances = int(max_instances) if max_instances else None

    max_prompt_length = int(gen_cfg.get("max_input_tokens", 2048))
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 512))
    temperature = float(gen_cfg.get("temperature", 1.0))
    top_p = float(gen_cfg.get("top_p", 0.9))
    do_sample = gen_cfg.get("do_sample")
    if do_sample is None:
        do_sample = temperature > 0
    else:
        do_sample = bool(do_sample)
    apply_chat_template = gen_cfg.get("apply_chat_template", True)
    single_turn_only = gen_cfg.get("single_turn_only", True)

    # Load tokenizer for prompt formatting
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    except Exception:
        if not tokenizer_fallback:
            raise
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_fallback, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load HH dataset and extract instructions
    raw_hh = load_dataset(dataset_repo, data_dir=data_dir, split=dataset_split)
    hh_ds = build_hh_dataset(raw_hh)
    if seed:
        hh_ds = hh_ds.shuffle(seed=seed)

    instructions = []
    raw_prompts = []
    for ex in hh_ds:
        prompt = ex.get("prompt", "")
        if not prompt:
            continue

        if single_turn_only:
            instr = extract_instruction_from_prompt(prompt)
            if instr is None:
                continue
            instruction_key = instr
            raw_prompt = format_prompt(tokenizer, instr, apply_chat_template)
        else:
            instruction_key = prompt
            if apply_chat_template and getattr(tokenizer, "apply_chat_template", None) and tokenizer.chat_template:
                messages = parse_hh_to_messages(prompt)
                raw_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                raw_prompt = prompt

        instructions.append(instruction_key)
        raw_prompts.append(raw_prompt)
        if max_instances and len(instructions) >= max_instances:
            break

    if not instructions:
        raise ValueError(
            "No HH examples found"
            if not single_turn_only
            else "No single-turn HH examples found"
        )

    mode_label = "single-turn" if single_turn_only else "multi-turn"
    print(f"Loaded {len(instructions)} {mode_label} instructions")

    # Build vLLM model
    llm = build_vllm_model(
        model_name,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        tokenizer=tokenizer_name,
    )

    sampling_params = get_sampling_params(
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
    )

    # Generate outputs
    records = []
    total_chunks = (len(raw_prompts) + chunk_size - 1) // chunk_size

    for start_idx, prompt_chunk in tqdm(
        chunked(raw_prompts, chunk_size),
        desc="Generating",
        total=total_chunks,
    ):
        try:
            outputs = llm.generate(prompt_chunk, sampling_params, use_tqdm=False)
        except TypeError:
            outputs = llm.generate(prompt_chunk, sampling_params)

        for i, out in enumerate(outputs):
            idx = start_idx + i
            text = out.outputs[0].text.lstrip()
            records.append(
                {
                    "instruction": instructions[idx],
                    "raw_instruction": raw_prompts[idx],
                    "output": text,
                    "generator": model_name,
                }
            )

    # Save outputs
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} outputs to {output_path}")


if __name__ == "__main__":
    main()
