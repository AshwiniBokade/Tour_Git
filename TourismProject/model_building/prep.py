import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import sys

# --- Config: change to the repo_id you choose and keep consistent ---
REPO_ID = "AshwiniBokade/Tourism-Project-Asgnmt"   # <-- ensure this is the one you want
REPO_TYPE = "dataset"

# --- Ensure token available ---
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("HF_TOKEN not set. Set it as an env var or GitHub Actions secret named HF_TOKEN.")

api = HfApi(token=HF_TOKEN)

# --- Load dataset from HF (if already present) or local path fallback ---
DATASET_PATH = f"hf://datasets/{REPO_ID}/tourism.csv"

try:
    df = pd.read_csv(DATASET_PATH)
    print("Loaded dataset from HF:", DATASET_PATH)
except Exception as e:
    # fallback to local file if hf load fails (local file must be present in the repo)
    print("Could not load from HF path:", e)
    local_path = "TourismProject/data/tourism.csv"
    if os.path.exists(local_path):
        print("Loading local dataset:", local_path)
        df = pd.read_csv(local_path)
    else:
        raise SystemExit(f"No dataset found at HF path and local file missing: {local_path}")

# --- Data cleaning ---
df.drop(columns=["CustomerID", "Unnamed: 0"], errors="ignore", inplace=True)
target_col = "ProdTaken"
df = df.dropna(subset=[target_col])

num_cols = df.select_dtypes(include=["number"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

for c in num_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].median())

for c in cat_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].mode()[0])

# --- Save cleaned and splits locally ---
os.makedirs("TourismProject/data", exist_ok=True)
df.to_csv("TourismProject/data/tourism_cleaned.csv", index=False)

X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# --- Ensure dataset repo exists on HF (create if missing) ---
try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Dataset repo '{REPO_ID}' exists.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{REPO_ID}' not found. Creating it now...")
    try:
        create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False, token=HF_TOKEN)
        print("Created dataset repo:", REPO_ID)
    except HfHubHTTPError as e:
        print("Failed to create dataset repo:", e)
        raise

# --- Upload the split files to the dataset repo ---
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for file_path in files:
    if not os.path.exists(file_path):
        raise SystemExit(f"File missing: {file_path}. Ensure training script created it.")
    print(f"Uploading {file_path} to {REPO_ID} ...")
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path),
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
    except HfHubHTTPError as e:
        print("Upload failed for", file_path, ":", e)
        # If unauthorized (401) or not found (404) show explicit guidance:
        if e.response is not None:
            code = e.response.status_code
            if code == 401:
                raise SystemExit("401 Unauthorized: check HF_TOKEN and token scopes (needs datasets write).")
            if code == 404:
                raise SystemExit("404 Not Found: repo does not exist or repo_id incorrect.")
        raise

print("All split files uploaded successfully.")
