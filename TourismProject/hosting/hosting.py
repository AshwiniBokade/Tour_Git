from huggingface_hub import HfApi
import os

# --- Auth ---
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("HF_TOKEN not set. Add it as GH secret and pass into the job.")

# --- Hugging Face Repo ID (Space) ---
HF_REPO_ID = os.environ.get("HF_REPO_ID") or "AshwiniBokade/Tourism-Project-Asgnmt"

# --- Local folder to deploy ---
local_folder = "TourismProject/deployment"

# --- Check ---
if not os.path.isdir(local_folder):
    print("ERROR: local_folder not found:", os.path.abspath(local_folder))
    print("Working dir:", os.getcwd())
    print("Top-level files:", os.listdir("."))
    raise SystemExit(1)

# --- Upload ---
print(f"Uploading folder '{local_folder}' to Hugging Face Space '{HF_REPO_ID}' (repo_type='space') ...")
api = HfApi(token=HF_TOKEN)
api.upload_folder(folder_path=local_folder, repo_id=HF_REPO_ID, repo_type="space")
print("✅ Upload completed.")
