from huggingface_hub import HfApi

repo_id = "Phainon/dpo_llama3_8b"
local_dir = "dpo_model"

api = HfApi()

api.create_repo(
    repo_id=repo_id,
    private=True,
    exist_ok=True,
)

api.upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    repo_type="model",
    commit_message="DPO fine-tuned LLaMA-3 8B (HH, seed=42)"
)

print("Uploaded model to HF:", repo_id)