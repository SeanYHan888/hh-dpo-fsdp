"""
Judge HF multi-model output files (beta vs beta) with GPT-5 mini.

Downloads output JSONs from a dataset repo, then runs pairwise judging
between beta checkpoints within each subfolder.
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from huggingface_hub import hf_hub_download
from openai import OpenAI

from judge_gpt4o import load_config
from judge_hf_multimodel import (
    group_files_by_subfolder,
    list_repo_files,
    run_pairwise_judge,
)


def resolve_beta_outputs(
    repo_id: str,
    subfolder_files: List[str],
    local_dir: Path,
    beta_regex: str,
) -> List[Tuple[str, Path]]:
    beta_re = re.compile(beta_regex)
    beta_paths: List[Tuple[str, Path]] = []

    for path in subfolder_files:
        filename = path.split("/")[-1]
        match = beta_re.match(filename)
        if not match:
            continue
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

    if not beta_paths:
        raise FileNotFoundError("No beta output files found in repo subfolder")

    beta_paths.sort(key=lambda item: _beta_sort_key(item[0]))
    return beta_paths


def _beta_sort_key(value: str):
    try:
        return float(value)
    except ValueError:
        return value


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_pairs(value: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for raw in _parse_csv(value):
        if ":" not in raw:
            raise ValueError("Pairs must be in the form 'beta_a:beta_b'")
        left, right = raw.split(":", 1)
        left = left.strip()
        right = right.strip()
        if not left or not right:
            raise ValueError("Pairs must be in the form 'beta_a:beta_b'")
        pairs.append((left, right))
    return pairs


def _all_pairs(values: Iterable[str]) -> List[Tuple[str, str]]:
    ordered = sorted(values, key=_beta_sort_key)
    pairs: List[Tuple[str, str]] = []
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            pairs.append((ordered[i], ordered[j]))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge HF multi-model outputs (beta vs beta) with GPT-5 mini"
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
        default="results/hf_judgments/beta_pairs",
        help="Output directory for results/summary files",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="outputs/hf_multi",
        help="Local download directory for HF files",
    )
    parser.add_argument(
        "--beta_regex",
        type=str,
        default=r"model_outputs_beta_(.+)\.json",
        help="Regex to identify beta output filenames",
    )
    parser.add_argument(
        "--betas",
        type=str,
        default=None,
        help="Optional comma-separated beta values to include",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Optional comma-separated beta pairs like '0.1:0.01,0.8:0.1'",
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

        beta_paths = resolve_beta_outputs(
            repo_id=args.repo_id,
            subfolder_files=subfolder_files,
            local_dir=local_dir,
            beta_regex=args.beta_regex,
        )

        beta_map: Dict[str, Path] = {value: path for value, path in beta_paths}

        if args.betas:
            requested = _parse_csv(args.betas)
            missing = [b for b in requested if b not in beta_map]
            if missing:
                raise ValueError(f"Missing beta outputs for: {', '.join(missing)}")
            beta_map = {b: beta_map[b] for b in requested}

        if args.pairs:
            pairs = _parse_pairs(args.pairs)
        else:
            pairs = _all_pairs(beta_map.keys())

        if not pairs:
            raise ValueError("No beta pairs to compare")

        summary_dir = output_dir / "summary" / subfolder

        for beta_a, beta_b in pairs:
            if beta_a not in beta_map or beta_b not in beta_map:
                raise ValueError(
                    f"Unknown beta pair '{beta_a}:{beta_b}'. "
                    "Use --betas or ensure outputs exist."
                )

            pair_name = f"{subfolder}_beta_{beta_a}_vs_beta_{beta_b}"
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
                model_a_path=beta_map[beta_a],
                model_b_path=beta_map[beta_b],
                model_a_name=f"beta_{beta_a}",
                model_b_name=f"beta_{beta_b}",
                results_path=results_path,
                summary_path=summary_path,
                max_instances=args.max_instances,
                resume=args.resume,
            )


if __name__ == "__main__":
    main()
