#!/usr/bin/env python3
"""Generate HH outputs for multiple beta models and upload to HF dataset repo.

This script:
1) Runs `generate_vllm.py` per model in an isolated subprocess.
2) Forces deterministic generation (`do_sample: false`).
3) Clears CUDA cache between model runs.
4) Uploads each JSON output to `do_sample_false/` (or chosen subfolder)
   in a Hugging Face dataset repo.
"""

from __future__ import annotations

import argparse
import copy
import gc
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi

DEFAULT_MODELS = [
    "W-61/hh-dpo-llama3.1-8b-fsdp-beta-0.001",
    "W-61/hh-dpo-llama3.1-8b-fsdp-beta-0.005",
    "W-61/hh-dpo-llama3.1-8b-fsdp-beta-0.008",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def beta_from_model_name(model_name: str) -> str:
    match = re.search(r"beta-([0-9.]+)$", model_name)
    if not match:
        raise ValueError(
            "Could not parse beta value from model name. "
            f"Expected suffix like '...-beta-0.005', got: {model_name}"
        )
    return match.group(1)


def clean_cache_between_runs() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        print(f"[WARN] CUDA cache cleanup skipped: {exc}")


def run_generation(
    python_exe: str,
    generate_script: Path,
    config_path: Path,
    model_name: str,
    output_path: Path,
    max_instances: int | None,
) -> None:
    cmd = [
        python_exe,
        str(generate_script),
        "--config",
        str(config_path),
        "--model_name",
        model_name,
        "--output_file",
        str(output_path),
    ]
    if max_instances is not None:
        cmd.extend(["--max_instances", str(max_instances)])

    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not output_path.exists():
        raise FileNotFoundError(f"Expected output file missing: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate outputs for multiple HH beta models with do_sample=false "
            "and upload them to a HF dataset repo subfolder."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Base YAML config (default: gpt_judge_hh_final/config.yaml)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="W-61/hh-dpo-multi-model-outputs-multi-turn",
        help="Target HF dataset repo id",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="do_sample_false",
        help="Subfolder path in dataset repo",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model ids to run",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/hf_multi/do_sample_false"),
        help="Local output directory",
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="Optional cap on number of examples",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create/use public dataset repo (default: private)",
    )
    parser.add_argument(
        "--skip_upload",
        action="store_true",
        help="Generate outputs only; do not upload",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    generate_script = script_dir / "generate_vllm.py"

    config_path = args.config or (script_dir / "config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not generate_script.exists():
        raise FileNotFoundError(f"generate_vllm.py not found: {generate_script}")

    base_cfg = load_yaml(config_path)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    if not args.skip_upload:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=not args.public,
            exist_ok=True,
        )

    with tempfile.TemporaryDirectory(prefix="hh_multi_gen_cfg_") as temp_dir:
        temp_root = Path(temp_dir)

        for idx, model_name in enumerate(args.models):
            beta = beta_from_model_name(model_name)
            output_name = f"model_outputs_beta_{beta}.json"
            output_path = output_dir / output_name

            run_cfg = copy.deepcopy(base_cfg)
            gen_cfg = run_cfg.setdefault("generation", {})
            gen_cfg["model_name"] = model_name
            gen_cfg["output_file"] = str(output_path)
            gen_cfg["do_sample"] = False
            gen_cfg["temperature"] = 0.0
            gen_cfg["top_p"] = 1.0

            run_cfg_path = temp_root / f"config_beta_{beta}.yaml"
            save_yaml(run_cfg_path, run_cfg)

            print(f"\n===== Generating beta={beta} from {model_name} =====")
            run_generation(
                python_exe=sys.executable,
                generate_script=generate_script,
                config_path=run_cfg_path,
                model_name=model_name,
                output_path=output_path,
                max_instances=args.max_instances,
            )

            if not args.skip_upload:
                path_in_repo = f"{args.subfolder}/{output_name}"
                print(f"[HF] Upload {output_path} -> {args.repo_id}:{path_in_repo}")
                api.upload_file(
                    path_or_fileobj=str(output_path),
                    path_in_repo=path_in_repo,
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    commit_message=(
                        f"Add {output_name} from {model_name} "
                        f"(do_sample=false, multi-turn HH)"
                    ),
                )

            if idx < len(args.models) - 1:
                print("[CLEANUP] Clearing cache before next model...")
                clean_cache_between_runs()

    print("\n[OK] Generation/upload complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
