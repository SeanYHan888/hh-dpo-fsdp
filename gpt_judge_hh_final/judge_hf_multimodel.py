"""
Judge HF multi-model output files (base vs betas) with GPT-5 mini.

Downloads output JSONs from a dataset repo, then runs pairwise judging
using the same logic as judge_gpt4o.py and writes results + summaries.
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

from huggingface_hub import HfApi, hf_hub_download
from openai import OpenAI
from tqdm import tqdm

from judge_gpt4o import call_gpt4o, load_config, load_outputs, parse_response


def list_repo_files(repo_id: str) -> List[str]:
    api = HfApi()
    return api.list_repo_files(repo_id, repo_type="dataset")


def group_files_by_subfolder(files: List[str], subfolders: List[str]) -> Dict[str, List[str]]:
    grouped = {name: [] for name in subfolders}
    for path in files:
        for name in subfolders:
            prefix = f"{name}/"
            if path.startswith(prefix):
                grouped[name].append(path)
                break
    return grouped


def resolve_outputs(
    repo_id: str,
    subfolder_files: List[str],
    local_dir: Path,
    base_name: str,
    beta_regex: str,
) -> Tuple[Path, List[Tuple[str, Path]]]:
    base_path = None
    beta_re = re.compile(beta_regex)
    beta_paths: List[Tuple[str, Path]] = []

    for path in subfolder_files:
        filename = path.split("/")[-1]
        if filename == base_name:
            base_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=path,
                    local_dir=str(local_dir),
                )
            )
            continue

        match = beta_re.match(filename)
        if match:
            beta_value = match.group(1)
            beta_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=path,
                    local_dir=str(local_dir),
                )
            )
            beta_paths.append((beta_value, beta_path))

    if base_path is None:
        raise FileNotFoundError(f"Base file '{base_name}' not found in repo subfolder")
    if not beta_paths:
        raise FileNotFoundError("No beta output files found in repo subfolder")

    beta_paths.sort(key=lambda item: item[0])
    return base_path, beta_paths


def run_pairwise_judge(
    client: OpenAI,
    prompt_template: str,
    model: str,
    temperature,
    max_tokens: int,
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
    system_prompt: str,
    seed: int,
    model_a_path: Path,
    model_b_path: Path,
    model_a_name: str,
    model_b_name: str,
    results_path: Path,
    summary_path: Path,
    max_instances: int,
    resume: bool,
) -> None:
    model_a_outputs, model_a_order = load_outputs(model_a_path)
    model_b_outputs, _ = load_outputs(model_b_path)

    common_instructions = [instr for instr in model_a_order if instr in model_b_outputs]
    if max_instances:
        common_instructions = common_instructions[:max_instances]

    print(f"Found {len(common_instructions)} common instructions to judge")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    counts = {model_a_name: 0, model_b_name: 0, "tie": 0}

    if resume and results_path.exists():
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

    rng = random.Random(seed)
    mode = "a" if resume else "w"

    with results_path.open(mode, encoding="utf-8") as f:
        for instruction in tqdm(pending, desc="Judging"):
            output_a = model_a_outputs[instruction]
            output_b = model_b_outputs[instruction]

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

    total = sum(counts.values())
    non_tie_total = total - counts.get("tie", 0)
    summary = {
        "total": total,
        "counts": counts,
        "win_rates": {k: v / total if total else 0.0 for k, v in counts.items()},
        "win_rates_exclude_tie": {
            model_a_name: counts[model_a_name] / non_tie_total if non_tie_total else 0.0,
            model_b_name: counts[model_b_name] / non_tie_total if non_tie_total else 0.0,
        },
        "model_a": model_a_name,
        "model_b": model_b_name,
        "judge_model": model,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {results_path}")
    print(f"Summary saved to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge HF multi-model outputs (base vs betas) with GPT-5 mini"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Config YAML path"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="W-61/hh-dpo-multi-model-outputs-multi-turn",
        help="HF dataset repo id",
    )
    parser.add_argument(
        "--subfolders",
        type=str,
        default="do_sample_true,do_sample_false",
        help="Comma-separated subfolders to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/hf_judgments",
        help="Output directory for results/summary files",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="outputs/hf_multi",
        help="Local download directory for HF files",
    )
    parser.add_argument(
        "--base_name",
        type=str,
        default="model_outputs_base.json",
        help="Base output filename",
    )
    parser.add_argument(
        "--beta_regex",
        type=str,
        default=r"model_outputs_beta_(.+)\.json",
        help="Regex to identify beta output filenames",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=None,
        help="Judge model name (overrides config)",
    )
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing results"
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    config = load_config(args.config)
    oracle_cfg = config.get("gpt4_oracle", {})

    prompt_template = oracle_cfg.get("prompt_template")
    if not prompt_template:
        raise ValueError("gpt4_oracle.prompt_template must be set in config")

    judge_model = args.judge_model or oracle_cfg.get("model") or "gpt-5-mini"
    temperature = oracle_cfg.get("temperature")
    max_tokens = oracle_cfg.get("max_tokens", 256)
    max_retries = oracle_cfg.get("max_retries", 5)
    initial_backoff = oracle_cfg.get("initial_backoff", 1.0)
    max_backoff = oracle_cfg.get("max_backoff", 60.0)
    system_prompt = oracle_cfg.get("system_prompt")
    seed = oracle_cfg.get("seed", 42)

    files = list_repo_files(args.repo_id)
    subfolders = [name.strip() for name in args.subfolders.split(",") if name.strip()]
    grouped = group_files_by_subfolder(files, subfolders)

    local_dir = Path(args.local_dir)
    output_dir = Path(args.output_dir)

    client = OpenAI()

    for subfolder in subfolders:
        subfolder_files = grouped.get(subfolder, [])
        if not subfolder_files:
            raise FileNotFoundError(f"No files found for subfolder '{subfolder}'")

        base_path, beta_paths = resolve_outputs(
            repo_id=args.repo_id,
            subfolder_files=subfolder_files,
            local_dir=local_dir,
            base_name=args.base_name,
            beta_regex=args.beta_regex,
        )

        summary_dir = output_dir / "summary" / subfolder
        for beta_value, beta_path in beta_paths:
            pair_name = f"{subfolder}_base_vs_beta_{beta_value}"
            results_path = output_dir / f"{pair_name}.jsonl"
            summary_path = summary_dir / f"summary_{pair_name}.json"

            print(f"\n=== Running {pair_name} ===")
            run_pairwise_judge(
                client=client,
                prompt_template=prompt_template,
                model=judge_model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                system_prompt=system_prompt,
                seed=seed,
                model_a_path=base_path,
                model_b_path=beta_path,
                model_a_name="base",
                model_b_name=f"beta_{beta_value}",
                results_path=results_path,
                summary_path=summary_path,
                max_instances=args.max_instances,
                resume=args.resume,
            )


if __name__ == "__main__":
    main()
