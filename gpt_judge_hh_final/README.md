# GPT-4o Judge Pipeline for HH-RLHF Evaluation

A self-contained pipeline for evaluating DPO/SFT models using GPT-4o as a pairwise judge on the Anthropic HH-RLHF dataset.

## Pipeline Overview

```
1. Extract HH Chosen Baseline  →  outputs/hh_chosen_outputs.json
2. Generate Model Outputs      →  outputs/model_outputs.json
3. GPT-4o Pairwise Judge       →  results/judgments.jsonl
                               →  results/summary.json
```

## Quick Start

### Prerequisites

```bash
pip install datasets transformers vllm openai pyyaml tqdm

export OPENAI_API_KEY="your-api-key"
```

### Step 1: Extract HH Chosen Baseline

```bash
python extract_chosen.py \
    --config config.yaml \
    --max_instances 500 \
    --output_file outputs/hh_chosen_outputs.json
```

### Step 2: Generate Model Outputs (vLLM)

Edit `config.yaml` to set your model:
```yaml
generation:
  model_name: "your-model-name-or-path"
```

Then run:
```bash
python generate_vllm.py \
    --config config.yaml \
    --output_file outputs/model_outputs_beta_0.01.json
```

### Step 3: Run GPT-4o Judge

```bash
python judge_gpt4o.py \
    --config config.yaml \
    --model_a outputs/model_outputs_beta_0.8.json \
    --model_b outputs/hh_chosen_outputs.json \
    --model_a_name "W-61/hh-dpo-llama3.1-8b-fsdp-beta-0.8" \
    --model_b_name "hh_chosen"
```

## HF Multi-Model Judge (Base vs Betas)

This script downloads multi-model output files from a Hugging Face dataset repo
and runs base-vs-beta comparisons for each subfolder.

### Install dependency

```bash
pip install huggingface_hub
```

### Run (default repo + subfolders)

```bash
python judge_hf_multimodel.py \
    --config config.yaml
```

### Outputs

Results go to:

```
results/hf_judgments/
  do_sample_true_base_vs_beta_0.01.jsonl
  do_sample_true_base_vs_beta_0.1.jsonl
  do_sample_true_base_vs_beta_0.8.jsonl
  do_sample_false_base_vs_beta_0.01.jsonl
  do_sample_false_base_vs_beta_0.1.jsonl
  do_sample_false_base_vs_beta_0.8.jsonl
  summary/
    do_sample_true/
      summary_do_sample_true_base_vs_beta_0.01.json
      summary_do_sample_true_base_vs_beta_0.1.json
      summary_do_sample_true_base_vs_beta_0.8.json
    do_sample_false/
      summary_do_sample_false_base_vs_beta_0.01.json
      summary_do_sample_false_base_vs_beta_0.1.json
      summary_do_sample_false_base_vs_beta_0.8.json
```

### Common options

```bash
python judge_hf_multimodel.py \
    --config config.yaml \
    --repo_id W-61/hh-dpo-multi-model-outputs \
    --subfolders do_sample_true,do_sample_false \
    --output_dir results/hf_judgments \
    --local_dir outputs/hf_multi \
    --max_instances 500 \
    --resume
```

## Files

| File | Description |
|------|-------------|
| `utils.py` | Shared HH dataset processing utilities |
| `extract_chosen.py` | Extract HH human-preferred responses |
| `generate_vllm.py` | Generate model outputs using vLLM |
| `judge_gpt4o.py` | Pairwise judging with GPT-4o |
| `judge_hf_multimodel.py` | Judge HF multi-model outputs (base vs betas) |
| `config.yaml` | Pipeline configuration |

## Output Format

### Model Outputs (JSON)
```json
[
  {
    "instruction": "How do I cook pasta?",
    "output": "Here's how...",
    "generator": "model-name"
  }
]
```

### Judgments (JSONL)
```json
{
  "instruction": "How do I cook pasta?",
  "winner": "A",
  "winner_key": "my_model",
  "comparison": "Response A is more detailed..."
}
```

### Summary (JSON)
```json
{
  "total": 500,
  "counts": {"my_model": 280, "hh_chosen": 180, "tie": 40},
  "win_rates": {"my_model": 0.56, "hh_chosen": 0.36, "tie": 0.08}
}
```

## Resume Interrupted Runs

```bash
python judge_gpt4o.py --config config.yaml --resume
```

## Configuration

Key settings in `config.yaml`:

```yaml
generation:
  model_name: "your-model"      # Model to evaluate
  max_instances: 500            # Number of examples
  do_sample: true              # Sampling on/off (false = greedy)
  temperature: 1.0              # Sampling temperature
  
gpt4_oracle:
  model: "gpt-4o-2024-08-06"    # Judge model
  prompt_template: |            # Evaluation prompt
    ...
```
