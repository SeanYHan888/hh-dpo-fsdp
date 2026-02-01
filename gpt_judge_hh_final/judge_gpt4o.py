"""
GPT-4o pairwise judge for comparing model outputs.

Compares two model outputs head-to-head using GPT-4o as judge.
Randomly shuffles A/B positions to mitigate position bias.
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError as exc:
    raise RuntimeError("openai package required: pip install openai") from exc


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_outputs(path: Path) -> Tuple[Dict[str, str], List[str]]:
    """Load model outputs and preserve instruction order."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")

    outputs = {}
    order = []
    for row in data:
        instruction = row.get("instruction")
        output = row.get("output")
        if not instruction or output is None:
            continue
        if instruction not in outputs:
            outputs[instruction] = output
            order.append(instruction)
    return outputs, order


def normalize_winner(value: str) -> Optional[str]:
    if not value:
        return None
    cleaned = value.strip().strip('"').strip().upper()
    return cleaned if cleaned in {"A", "B", "TIE"} else None


def parse_response(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse GPT-4o response for comparison and winner."""
    text = text.strip()
    if not text:
        return None, None

    comparison = None
    winner = None

    # Try JSON format
    if text.startswith("{"):
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                comparison = payload.get("comparison") or payload.get("Comparison")
                winner = normalize_winner(
                    str(payload.get("winner") or payload.get("Winner") or "")
                )
        except json.JSONDecodeError:
            pass

    # Try line-based format
    for line in text.splitlines():
        line = line.strip()
        lower = line.lower()
        if lower.startswith("comparison:"):
            comparison = line.split(":", 1)[1].strip()
        elif lower.startswith("winner:"):
            winner = normalize_winner(line.split(":", 1)[1])

    # Fallback regex
    if winner is None:
        match = re.search(r'"winner"\s*:\s*"(A|B|TIE)"', text, re.IGNORECASE)
        if match:
            winner = match.group(1).upper()

    if winner is None:
        match = re.search(r"\b(A|B|TIE)\b", text, re.IGNORECASE)
        if match:
            winner = match.group(1).upper()

    return comparison, winner


def call_gpt4o(
    client: OpenAI,
    prompt: str,
    model: str,
    temperature: Optional[float],
    max_tokens: int,
    max_retries: int = 5,
    initial_backoff: float = 1.0,
    max_backoff: float = 60.0,
    system_prompt: Optional[str] = None,
) -> Tuple[str, dict]:
    """Call GPT-4o with retry logic."""
    attempt = 0
    backoff = initial_backoff
    last_error = None

    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    while attempt <= max_retries:
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature

            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or ""
            usage = {}
            if response.usage:
                usage = (
                    response.usage.model_dump()
                    if hasattr(response.usage, "model_dump")
                    else dict(response.usage)
                )
            return text, usage
        except Exception as exc:
            last_error = exc
            attempt += 1
            if attempt > max_retries:
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

    raise RuntimeError(f"OpenAI API failed after {max_retries} retries") from last_error


def main():
    parser = argparse.ArgumentParser(
        description="Pairwise judge model outputs with GPT-4o"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Config YAML path"
    )
    parser.add_argument("--model_a", type=str, help="Path to model A outputs JSON")
    parser.add_argument("--model_b", type=str, help="Path to model B outputs JSON")
    parser.add_argument(
        "--model_a_name", type=str, default="model_a", help="Model A label"
    )
    parser.add_argument(
        "--model_b_name", type=str, default="model_b", help="Model B label"
    )
    parser.add_argument("--results_file", type=str, help="Override results output path")
    parser.add_argument("--summary_file", type=str, help="Override summary output path")
    parser.add_argument("--max_instances", type=int, help="Max examples to judge")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing results"
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    config = load_config(args.config)
    oracle_cfg = config.get("gpt4_oracle", {})
    output_cfg = config.get("output", {})
    inputs_cfg = config.get("inputs", {})

    # Get prompt template
    prompt_template = oracle_cfg.get("prompt_template")
    if not prompt_template:
        raise ValueError("gpt4_oracle.prompt_template must be set in config")

    # Load model outputs
    model_a_path = Path(args.model_a or inputs_cfg.get("model_a", ""))
    model_b_path = Path(args.model_b or inputs_cfg.get("model_b", ""))

    if not model_a_path.exists() or not model_b_path.exists():
        raise FileNotFoundError("Both model_a and model_b output files required")

    model_a_outputs, model_a_order = load_outputs(model_a_path)
    model_b_outputs, model_b_order = load_outputs(model_b_path)

    model_a_name = args.model_a_name or inputs_cfg.get("model_a_name", "model_a")
    model_b_name = args.model_b_name or inputs_cfg.get("model_b_name", "model_b")

    # Find common instructions
    common_instructions = [instr for instr in model_a_order if instr in model_b_outputs]

    max_instances = args.max_instances or oracle_cfg.get("max_instances")
    if max_instances:
        common_instructions = common_instructions[:max_instances]

    print(f"Found {len(common_instructions)} common instructions to judge")

    # Output paths
    results_path = Path(
        args.results_file or output_cfg.get("results_file", "results/judgments.jsonl")
    )
    summary_path = Path(
        args.summary_file or output_cfg.get("summary_file", "results/summary.json")
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Judge settings
    model = oracle_cfg.get("model", "gpt-4o-2024-08-06")
    temperature = oracle_cfg.get("temperature")
    max_tokens = oracle_cfg.get("max_tokens", 256)
    max_retries = oracle_cfg.get("max_retries", 5)
    initial_backoff = oracle_cfg.get("initial_backoff", 1.0)
    max_backoff = oracle_cfg.get("max_backoff", 60.0)
    system_prompt = oracle_cfg.get("system_prompt")
    seed = oracle_cfg.get("seed", 42)

    # Resume logic
    seen = set()
    counts = {model_a_name: 0, model_b_name: 0, "tie": 0}

    if args.resume and results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    seen.add(row.get("instruction"))
                    winner_key = row.get("winner_key")
                    if winner_key == model_a_name:
                        counts[model_a_name] += 1
                    elif winner_key == model_b_name:
                        counts[model_b_name] += 1
                    else:
                        counts["tie"] += 1
                except json.JSONDecodeError:
                    continue

    pending = [instr for instr in common_instructions if instr not in seen]
    print(f"Judging {len(pending)} examples (skipped {len(seen)} already done)")

    client = OpenAI()
    rng = random.Random(seed)

    mode = "a" if args.resume else "w"
    with results_path.open(mode, encoding="utf-8") as f:
        for instruction in tqdm(pending, desc="Judging"):
            output_a = model_a_outputs[instruction]
            output_b = model_b_outputs[instruction]

            # Randomize position to mitigate bias
            outputs = [(model_a_name, output_a), (model_b_name, output_b)]
            rng.shuffle(outputs)
            label_map = {"A": outputs[0][0], "B": outputs[1][0]}

            prompt = prompt_template.format(
                instruction=instruction,
                output_a=outputs[0][1],
                output_b=outputs[1][1],
            )

            content, usage = call_gpt4o(
                client=client,
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                system_prompt=system_prompt,
            )

            comparison, winner = parse_response(content)
            if winner is None:
                winner = "TIE"

            winner_key = label_map.get(winner, "tie")
            if winner_key == model_a_name:
                counts[model_a_name] += 1
            elif winner_key == model_b_name:
                counts[model_b_name] += 1
            else:
                counts["tie"] += 1

            result = {
                "instruction": instruction,
                "comparison": comparison,
                "winner": winner,
                "winner_key": winner_key,
                "labels": label_map,
                "response_a": outputs[0][1],
                "response_b": outputs[1][1],
                "model": model,
                "raw_response": content,
                "usage": usage,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

    # Write summary
    total = sum(counts.values())
    summary = {
        "total": total,
        "counts": counts,
        "win_rates": {k: v / total if total else 0.0 for k, v in counts.items()},
        "model_a": model_a_name,
        "model_b": model_b_name,
        "judge_model": model,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {results_path}")
    print(f"Summary saved to {summary_path}")
    print(f"\nSummary:")
    print(
        f"  {model_a_name} wins: {counts[model_a_name]} ({counts[model_a_name] / total * 100:.1f}%)"
    )
    print(
        f"  {model_b_name} wins: {counts[model_b_name]} ({counts[model_b_name] / total * 100:.1f}%)"
    )
    print(f"  Ties: {counts['tie']} ({counts['tie'] / total * 100:.1f}%)")


if __name__ == "__main__":
    main()
