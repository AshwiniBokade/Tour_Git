from huggingface_hub import HfApi
import os
import sys

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("HF_TOKEN not set. Add it as GH secret and pass into the job.")

# Use your HF repo id directly (replace if you keep config elsewhere)
HF_REPO_ID = "AshwiniBokade/Your-HF-Space-Repo-Here"  # e.g. "AshwiniBokade/tourism-space"

api = HfApi(token=HF_TOKEN)

local_folder = "TourismProject/deployment"  # folder to upload
if not os.path.isdir(local_folder):
    print("ERROR: local_folder not found:", os.path.abspath(local_folder))
    print("Working dir:", os.getcwd())
    print("Top-level files:", os.listdir("."))
    raise SystemExit(1)

print(f"Uploading folder '{local_folder}' to Hugging Face Space '{HF_REPO_ID}' ...")
api.upload_folder(
    folder_path=local_folder,
    repo_id=HF_REPO_ID,
    repo_type="space",
)
print("Upload completed.")
