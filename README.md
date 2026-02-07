# FSDP DPO Training on Anthropic HH-RLHF

This repository trains a DPO policy model with PyTorch FSDP, evaluates it against HH chosen responses, and supports GPT-based pairwise judging.

Main flow:

1. Train DPO with FSDP (`training.py`)
2. Generate policy/base responses with vLLM (`test_og/generate_response.py`)
3. Build pairwise comparison files (`test_og/generate_pairwise.py`)
4. Judge with OpenAI Responses API (`test_og/gpt_judge_pairwise.py`)
5. Summarize win rate (`test_og/summarize_judged.py`)

## Repository Layout

- `training.py`: distributed FSDP DPO training loop.
- `process_hh_dataset.py`: HH parsing, prompt/response splitting, chat template conversion, dataloaders.
- `dpo_loss.py`: DPO loss + margin/tail logging helpers.
- `compute_batch_log_prob.py`: policy/ref batch log-prob computation.
- `threshold.py`: warmup quantile + EMA threshold update.
- `test_og/generate_response.py`: vLLM generation for policy and base models.
- `test_og/generate_pairwise.py`: creates A/B pairwise JSONL for judging.
- `test_og/gpt_judge_pairwise.py`: GPT judge (OpenAI API) over pairwise JSONL.
- `test_og/summarize_judged.py`: win/loss/tie + Wilson CI summary.
- `upload_hf.py`: simple HF model upload script (currently hardcoded values).
- `config.yaml`: training/eval configuration.

## Prerequisites

- Python `>=3.10`
- NVIDIA GPUs + CUDA for training (`torchrun` + NCCL)
- Access to model repo used in config (default is gated Llama model)
- Linux for `vllm` (in `pyproject.toml`, `vllm` is Linux-only)

Environment variables commonly needed:

```bash

export HF_TOKEN=...

export WANDB_API_KEY=...

export OPENAI_API_KEY=...

```

If using gated Meta Llama models, make sure you are authenticated with Hugging Face and have access.

## Installation

Option A (`uv`):

```bash

uv sync

```

Option B (`pip`):

```bash

pip install -r requirements.txt

```

Install a CUDA-enabled PyTorch build matching your system if needed.

## Configuration (`config.yaml`)

Main sections:

- `policy_model`, `ref_model`, `precision`
- `dataset`:
  - `dataset_name` (default `Anthropic/hh-rlhf`)
  - `subset` (HF split slice, e.g. `train[:100%]`)
  - `val_ratio`, `max_len`, `use_chat_template`

- `dpo_training`:
  - `epochs`, `batch_size`, `learning_rate`, `dpo_beta`
  - `warmup_steps`, `max_grad_norm`, `save_dir`

- `tail_test`: threshold/tail logging settings and log dir
- `HH_test`: generation/eval sampling config and output jsonl paths
- `vllm`: generation parallelism/chunking
- `huggingface`: repo names (not consumed by `upload_hf.py` yet)

## How To Use

### 1) Train (FSDP DPO)

Training uses distributed initialization unconditionally, so run via `torchrun` (even on one GPU):

```bash

torchrun --nproc_per_node=1 training.py --config config.yaml

```

Multi-GPU example:

```bash

torchrun --nproc_per_node=4 training.py --config config.yaml

```

Outputs:

- model checkpoint + tokenizer in `dpo_training.save_dir` (default `dpo_model/`)
- margin/tail logs in `tail_test.log_dir` (default `logs/margins/`)

### 2) Generate Policy/Base Responses

```bash

python test_og/generate_response.py --config config.yaml

```

Outputs (from `HH_test` config):

- `HH_test_policy_out` (default `eval_outputs/policy_out.jsonl`)
- `HH_test_base_out` (default `eval_outputs/base_out.jsonl`)

### 3) Convert to Pairwise A/B Files

Policy vs HH chosen:

```bash

python test_og/generate_pairwise.py \

  --in_file eval_outputs/policy_out.jsonl \

  --out_file eval_outputs/policy_pairwise.jsonl

```

Base vs HH chosen:

```bash

python test_og/generate_pairwise.py \

  --in_file eval_outputs/base_out.jsonl \

  --out_file eval_outputs/base_pairwise.jsonl

```

If you want randomized A/B positions, add `--shuffle_ab` (then interpret results with `winner_model_vs_chosen` from judge output, not just raw `winner`).

### 4) Judge with GPT

```bash

python test_og/gpt_judge_pairwise.py \

  --in_file eval_outputs/policy_pairwise.jsonl \

  --out_file eval_outputs/policy_judged.jsonl \

  --model gpt-4o-mini

```

Repeat for `base_pairwise.jsonl` if needed.

### 5) Summarize Judging

```bash

python test_og/summarize_judged.py --in_file eval_outputs/policy_judged.jsonl

```

This script summarizes raw `winner` (`a`/`b`/`tie`), so it is most direct when A/B was not shuffled.

## Upload to Hugging Face

`upload_hf.py` currently uses hardcoded values:

- `repo_id = "Phainon/dpo_llama3_8b"`
- `local_dir = "dpo_model"`

Update those values in `upload_hf.py`, then run:

```bash

python upload_hf.py

```

## Notes and Constraints

- Current FSDP wrapping policy targets `LlamaDecoderLayer`; this code is tuned for Llama-family architectures.
- `training.py` wraps both policy and reference model with FSDP and logs only from rank 0.
- `test_chat_template.py` can be used to validate HH -> chat-template conversion.
- Additional judging utilities exist in `gpt_judge_hh_final/`.
