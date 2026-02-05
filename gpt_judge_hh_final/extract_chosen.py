"""
Extract HH chosen responses as baseline for evaluation.

Outputs JSON in same format as model generation scripts for fair comparison.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import yaml
from datasets import load_dataset

from utils import (
    build_hh_dataset,
    extract_instruction_from_prompt,
    extract_single_turn_instruction,
    parse_hh_to_messages,
)


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def extract_single_turn_pair(text: str) -> Optional[tuple]:
    """Extract (instruction, response) from single-turn HH conversations."""
    messages = parse_hh_to_messages(text)
    if len(messages) != 2:
        return None
    if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return None
    return messages[0]["content"], messages[1]["content"]


def normalize_hh_prompt_key(prompt: str) -> str:
    """Match the mixed instruction-key format used by model outputs.

    - Strip leading whitespace to drop leading \\n\\n.
    - If it's single-turn HH prompt, return just the user text.
    - Otherwise return the full prompt (without leading whitespace).
    """
    prompt = str(prompt).replace("\r\n", "\n").replace("\r", "\n").lstrip()
    if not prompt:
        return prompt

    # extract_instruction_from_prompt expects leading \\n\\n
    detect = prompt if prompt.startswith("\n\n") else "\n\n" + prompt
    single_turn = extract_instruction_from_prompt(detect)
    return single_turn if single_turn is not None else prompt


def main():
    parser = argparse.ArgumentParser(
        description="Extract HH chosen responses as evaluation baseline"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSON path (overrides config)",
    )
    parser.add_argument(
        "--max_instances", type=int, default=None, help="Max examples to extract"
    )
    parser.add_argument(
        "--dataset_split", type=str, default="test", help="Dataset split to use"
    )
    parser.add_argument(
        "--use_raw_hh",
        action="store_true",
        help="Use build_hh_dataset for raw HH processing",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    gen_cfg = config.get("generation", {})

    dataset_repo = gen_cfg.get("dataset_repo", "Anthropic/hh-rlhf")
    data_dir = gen_cfg.get("dataset_data_dir", "harmless-base")
    split = args.dataset_split or gen_cfg.get("dataset_split", "test")
    max_instances = args.max_instances or gen_cfg.get("max_instances")
    seed = gen_cfg.get("seed", 42)
    single_turn_only = gen_cfg.get("single_turn_only", True)
    output_file = args.output_file or gen_cfg.get(
        "chosen_output_file", "outputs/hh_chosen_outputs.json"
    )

    outputs = []
    use_raw_hh = args.use_raw_hh or not single_turn_only

    if use_raw_hh:
        # Use build_hh_dataset for raw HH triplet processing (required for multi-turn)
        raw_hh = load_dataset(dataset_repo, data_dir=data_dir, split=split)
        hh_ds = build_hh_dataset(raw_hh).shuffle(seed=seed)
        if max_instances:
            hh_ds = hh_ds.select(range(min(max_instances, len(hh_ds))))

        for row in hh_ds:
            prompt = row.get("prompt", "")
            if not prompt:
                continue

            if single_turn_only:
                # For raw HH, we need to parse the prompt to get single-turn instruction
                messages = parse_hh_to_messages(prompt)
                if len(messages) == 1 and messages[0]["role"] == "user":
                    instruction = messages[0]["content"]
                else:
                    continue
            else:
                instruction = normalize_hh_prompt_key(prompt)

            response = str(row["chosen"]).lstrip()
            outputs.append(
                {
                    "instruction": instruction,
                    "output": response,
                    "generator": f"{dataset_repo}:chosen",
                }
            )
    else:
        # Direct extraction from HH dataset
        dataset = load_dataset(dataset_repo, data_dir=data_dir, split=split)
        skipped = 0

        for row in dataset:
            chosen_text = row.get("chosen")
            if not chosen_text:
                skipped += 1
                continue

            pair = extract_single_turn_pair(chosen_text)
            if pair is None:
                skipped += 1
                continue

            instruction, response = pair
            outputs.append(
                {
                    "instruction": instruction,
                    "output": response,
                    "generator": f"{dataset_repo}:chosen",
                }
            )

            if max_instances and len(outputs) >= max_instances:
                break

        print(f"Skipped {skipped} non-single-turn examples")

    # Write outputs
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(outputs)} chosen outputs to {output_path}")


if __name__ == "__main__":
    main()
