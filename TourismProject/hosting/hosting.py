
from huggingface_hub import HfApi
import os
from TourismProject.deployment.config import HF_REPO_ID

api = HfApi(token=os.environ.get("HF_TOKEN"))
api.upload_folder(
    folder_path="TourismProject/deployment",   #local folder containing your files
    repo_id=HF_REPO_ID,                         # the target repo, the repo id is case sensitive
    repo_type="space",                          # dataset, mode or space
)
