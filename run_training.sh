#!/bin/bash
set -e

# Configuration
CONFIG_FILE="${1:-config.yaml}"
NUM_GPUS="${2:-1}"

echo "=================================="
echo "DPO Training Runner"
echo "=================================="
echo "Config: $CONFIG_FILE"
echo "GPUs: $NUM_GPUS"
echo "=================================="

# Sync dependencies
echo "[1/4] Syncing dependencies..."
uv sync

# Activate virtual environment
source .venv/bin/activate

# Run distributed training
echo "[2/4] Starting DPO training..."
uv run torchrun --nproc_per_node=$NUM_GPUS training.py --config $CONFIG_FILE

echo "[3/4] Training complete!"

# Extract paths from config
LOG_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['tail_test']['log_dir'])")
MODEL_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['dpo_training']['save_dir'])")
LOGS_REPO=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['huggingface']['logs_repo'])")
MODEL_REPO=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['huggingface']['model_repo'])")

# Auto-upload logs to HF
echo "[4/4] Uploading training logs to HuggingFace..."
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='$LOG_DIR/global_margins_log.jsonl',
    path_in_repo='global_margins_log.jsonl',
    repo_id='$LOGS_REPO',
    repo_type='dataset',
    create_pr=False
)
api.upload_file(
    path_or_fileobj='$LOG_DIR/global_tail_test_and_beta_log.jsonl',
    path_in_repo='global_tail_test_and_beta_log.jsonl',
    repo_id='$LOGS_REPO',
    repo_type='dataset',
    create_pr=False
)
print('✓ Logs uploaded to $LOGS_REPO')
"

# Ask user about model upload
echo ""
read -p "Upload trained model to HuggingFace? ($MODEL_REPO) [y/N]: " upload_model
if [[ "$upload_model" =~ ^[Yy]$ ]]; then
    echo "Uploading model to $MODEL_REPO..."
    python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='$MODEL_DIR',
    repo_id='$MODEL_REPO',
    repo_type='model'
)
print('✓ Model uploaded to $MODEL_REPO')
"
else
    echo "Skipping model upload."
fi

echo "=================================="
echo "All done!"
echo "=================================="
