#!/usr/bin/env python3
"""Run HH harmless DPO beta sweep on 4 GPUs and upload artifacts to Hugging Face.

This script runs `training.py` twice with betas 0.005 and 0.008, logs margin files,
and uploads:
1) trained model checkpoint folder to a HF model repo
2) margin log JSONL files to a HF dataset repo

Expected model naming:
- W-61/hh-dpo-llama3.1-8b-fsdp-beta-0.005
- W-61/hh-dpo-llama3.1-8b-fsdp-beta-0.008
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi

BETAS = [0.005, 0.008]
MARGIN_FILES = [
    "global_margins_log.jsonl",
    "global_tail_test_and_beta_log.jsonl",
]


def format_beta(beta: float) -> str:
    return f"{beta:.3f}"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def clean_cache_between_runs() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        print(f"[WARN] CUDA cache cleanup skipped: {exc}")


def build_run_config(base_cfg: dict[str, Any], beta: float, hf_org: str, run_root: Path) -> dict[str, Any]:
    beta_str = format_beta(beta)
    cfg = copy.deepcopy(base_cfg)

    cfg.setdefault("dataset", {})
    cfg["dataset"]["dataset_name"] = "Anthropic/hh-rlhf"

    cfg.setdefault("HH_test", {})
    cfg["HH_test"]["hh_test"] = "Anthropic/hh-rlhf"

    cfg.setdefault("dpo_training", {})
    cfg["dpo_training"]["dpo_beta"] = float(beta_str)
    cfg["dpo_training"]["save_dir"] = (run_root / "dpo_model").as_posix()

    cfg.setdefault("tail_test", {})
    cfg["tail_test"]["log_dir"] = (run_root / "logs" / "margins").as_posix()

    cfg["run_name"] = f"hh-dpo-llama3.1-8b-fsdp-beta-{beta_str}"

    cfg.setdefault("huggingface", {})
    cfg["huggingface"]["model_repo"] = f"{hf_org}/hh-dpo-llama3.1-8b-fsdp-beta-{beta_str}"
    cfg["huggingface"]["logs_repo"] = f"{hf_org}/hh-dpo-logs-beta-{beta_str}"

    return cfg


def run_training(repo_root: Path, config_path: Path, nproc_per_node: int) -> None:
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        "training.py",
        "--config",
        str(config_path),
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)


def upload_model(api: HfApi, save_dir: Path, repo_id: str, private: bool) -> None:
    if not save_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {save_dir}")

    print(f"[HF] upload model -> {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(save_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload DPO checkpoint for {save_dir.parent.name}",
    )


def upload_margin_logs(api: HfApi, log_dir: Path, repo_id: str, private: bool) -> None:
    print(f"[HF] upload margin logs -> {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    for filename in MARGIN_FILES:
        src = log_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Required margin log missing: {src}")

        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload {filename}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run HH harmless beta sweep (0.005, 0.008) and upload to HF."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Base YAML config path",
    )
    parser.add_argument(
        "--hf_org",
        type=str,
        default="W-61",
        help="Hugging Face org/user prefix",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=4,
        help="Number of GPUs for torchrun (default: 4)",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("sweep_runs"),
        help="Root dir for per-beta configs/artifacts",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create public HF repos (default: private)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Write configs and print commands only; do not train/upload",
    )
    args = parser.parse_args()

    hf_org = args.hf_org.strip()
    if not hf_org:
        raise ValueError("--hf_org cannot be empty")

    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else (repo_root / args.config)
    base_cfg = load_yaml(config_path)

    api = HfApi()
    private = not args.public

    sweep_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = repo_root / args.output_root / f"hh_beta_sweep_{sweep_stamp}"
    sweep_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "base_config": str(config_path.resolve()),
        "hf_org": hf_org,
        "nproc_per_node": args.nproc_per_node,
        "dry_run": args.dry_run,
        "runs": [],
    }

    for idx, beta in enumerate(BETAS):
        beta_str = format_beta(beta)
        run_root = sweep_root / f"beta_{beta_str}"

        run_cfg = build_run_config(base_cfg=base_cfg, beta=beta, hf_org=hf_org, run_root=run_root)
        run_cfg_path = run_root / f"config_beta_{beta_str}.yaml"
        save_yaml(run_cfg_path, run_cfg)

        save_dir = Path(run_cfg["dpo_training"]["save_dir"])
        log_dir = Path(run_cfg["tail_test"]["log_dir"])
        model_repo = run_cfg["huggingface"]["model_repo"]
        logs_repo = run_cfg["huggingface"]["logs_repo"]

        run_record: dict[str, Any] = {
            "beta": beta_str,
            "config": str(run_cfg_path),
            "save_dir": str(save_dir),
            "log_dir": str(log_dir),
            "model_repo": model_repo,
            "logs_repo": logs_repo,
            "status": "pending",
        }

        try:
            print(f"\n===== beta={beta_str} =====")
            if args.dry_run:
                print(
                    "[DRY-RUN] torchrun",
                    f"--nproc_per_node={args.nproc_per_node}",
                    "training.py --config",
                    run_cfg_path,
                )
                run_record["status"] = "dry_run"
            else:
                run_training(repo_root=repo_root, config_path=run_cfg_path, nproc_per_node=args.nproc_per_node)
                upload_model(api=api, save_dir=save_dir, repo_id=model_repo, private=private)
                upload_margin_logs(api=api, log_dir=log_dir, repo_id=logs_repo, private=private)
                run_record["status"] = "success"
        except Exception as exc:
            run_record["status"] = "failed"
            run_record["error"] = str(exc)
            summary["runs"].append(run_record)

            summary_path = sweep_root / "sweep_summary.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            print(f"[ERROR] beta={beta_str} failed: {exc}", file=sys.stderr)
            return 1

        summary["runs"].append(run_record)
        if idx < len(BETAS) - 1:
            clean_cache_between_runs()

    summary_path = sweep_root / "sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Sweep finished. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
