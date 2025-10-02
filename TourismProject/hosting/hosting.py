from huggingface_hub import HfApi
import os, sys

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("HF_TOKEN not set. Add it as GH secret and pass into the job.")

# Either set HF_REPO_ID directly here or read from a config file if you made it importable
HF_REPO_ID = os.environ.get("HF_REPO_ID") or "AshwiniBokade/tourism-space-example"  # <- replace with your space id

api = HfApi(token=HF_TOKEN)

local_folder = "TourismProject/deployment"
if not os.path.isdir(local_folder):
    print("ERROR: local_folder not found:", os.path.abspath(local_folder))
    print("Working dir:", os.getcwd())
    print("Top-level files:", os.listdir("."))
    raise SystemExit(1)

print(f"Uploading folder '{local_folder}' to Hugging Face Space '{HF_REPO_ID}' (repo_type='space') ...")
api.upload_folder(folder_path=local_folder, repo_id=HF_REPO_ID, repo_type="space")
print("Upload completed.")
