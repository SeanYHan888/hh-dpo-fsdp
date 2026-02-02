#!/usr/bin/env python3
"""
DPO Training Script for Llama-3.1-8B-Instruct with FSDP

This script:
1. Runs DPO training with beta=0.01 on the Anthropic HH-RLHF dataset
2. Uses FSDP for distributed training across multiple GPUs
3. Uploads the trained model and logs to Hugging Face

Usage:
    # Single GPU (not recommended for 8B model)
    python run_dpo_training.py

    # Multi-GPU with torchrun (recommended)
    torchrun --nproc_per_node=4 training.py --config config.yaml

    # Or use this wrapper script
    ./run_dpo_training.py
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def get_gpu_count():
    """Get the number of available GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            return len(lines)
    except FileNotFoundError:
        pass
    return 0


def run_training(config_path: str, num_gpus: int = None):
    """Run the DPO training with torchrun."""
    if num_gpus is None:
        num_gpus = get_gpu_count()
        if num_gpus == 0:
            print("Error: No GPUs detected. DPO training requires GPU(s).")
            sys.exit(1)

    print(f"Starting DPO training with {num_gpus} GPU(s)...")
    print(f"Config: {config_path}")
    print("-" * 60)

    # Get the directory where training.py is located
    script_dir = Path(__file__).parent.resolve()
    training_script = script_dir / "training.py"

    if not training_script.exists():
        print(f"Error: training.py not found at {training_script}")
        sys.exit(1)

    # Build the torchrun command
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        str(training_script),
        "--config",
        config_path,
    ]

    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)

    # Run training
    result = subprocess.run(cmd, cwd=str(script_dir))
    return result.returncode


def upload_to_huggingface(config_path: str):
    """Upload the trained model and logs to Hugging Face."""
    import yaml
    from huggingface_hub import HfApi, login

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    hf_config = config.get("huggingface", {})
    model_repo = hf_config.get("model_repo", "W-61/hh-dpo-llama3.1-8b-fsdp-beta-0.01")
    logs_repo = hf_config.get("logs_repo", "W-61/hh-dpo-logs-beta-0.01")

    save_dir = config["dpo_training"]["save_dir"]
    log_dir = config["tail_test"]["log_dir"]

    api = HfApi()

    # Check HF authentication
    try:
        api.whoami()
        print("Authenticated with Hugging Face")
    except Exception:
        print(
            "Not logged in to Hugging Face. Please run 'huggingface-cli login' first."
        )
        sys.exit(1)

    # Upload model
    if os.path.exists(save_dir):
        print(f"\nUploading model to {model_repo}...")
        try:
            api.create_repo(repo_id=model_repo, private=False, exist_ok=True)
            api.upload_folder(
                folder_path=save_dir,
                repo_id=model_repo,
                repo_type="model",
                commit_message=f"DPO fine-tuned Llama-3.1-8B-Instruct (beta=0.01, HH-RLHF)",
            )
            print(f"✓ Model uploaded to https://huggingface.co/{model_repo}")
        except Exception as e:
            print(f"Failed to upload model: {e}")
    else:
        print(f"Warning: Model directory {save_dir} not found. Skipping model upload.")

    # Upload logs
    if os.path.exists(log_dir):
        print(f"\nUploading logs to {logs_repo}...")
        try:
            api.create_repo(
                repo_id=logs_repo, repo_type="dataset", private=False, exist_ok=True
            )
            api.upload_folder(
                folder_path=log_dir,
                repo_id=logs_repo,
                repo_type="dataset",
                commit_message=f"Training logs for DPO beta=0.01",
            )
            print(f"✓ Logs uploaded to https://huggingface.co/datasets/{logs_repo}")
        except Exception as e:
            print(f"Failed to upload logs: {e}")
    else:
        print(f"Warning: Log directory {log_dir} not found. Skipping log upload.")


def main():
    parser = argparse.ArgumentParser(
        description="DPO Training Script for Llama-3.1-8B-Instruct with FSDP"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: auto-detect)",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and only upload existing model",
    )
    parser.add_argument(
        "--skip_upload",
        action="store_true",
        help="Skip uploading to Hugging Face",
    )
    args = parser.parse_args()

    # Get absolute path to config
    script_dir = Path(__file__).parent.resolve()
    config_path = script_dir / args.config
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    print("=" * 60)
    print("DPO Training for Llama-3.1-8B-Instruct")
    print("=" * 60)
    print(f"Beta: 0.01")
    print(f"Target HF Repo: W-61/hh-dpo-llama3.1-8b-fsdp-beta-0.01")
    print("=" * 60)

    # Run training
    if not args.skip_training:
        return_code = run_training(str(config_path), args.num_gpus)
        if return_code != 0:
            print(f"\nTraining failed with return code {return_code}")
            sys.exit(return_code)
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)

    # Upload to Hugging Face
    if not args.skip_upload:
        print("\n" + "=" * 60)
        print("Uploading to Hugging Face...")
        print("=" * 60)
        upload_to_huggingface(str(config_path))
        print("\n" + "=" * 60)
        print("Upload completed!")
        print("=" * 60)

    print("\nDone!")


if __name__ == "__main__":
    main()
