import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# === CONFIG ===
REPO_ID = "AshwiniBokade/Tourism-Project-Asgnmt"  # your correct dataset id
REPO_TYPE = "dataset"
LOCAL_FALLBACK = "TourismProject/data/tourism.csv"  # local fallback path if HF load fails
OUTPUT_DIR = "TourismProject/data"  # where to save cleaned file locally

# === HF TOKEN ===
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable not found. Set it as a secret or export locally.")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

# === Helper: try to load dataset from HF, else local ===
hf_path = f"hf://datasets/{REPO_ID}/tourism.csv"
df = None
try:
    print(f"Trying to load dataset from Hugging Face: {hf_path}")
    df = pd.read_csv(hf_path)
    print("Loaded dataset from Hugging Face successfully.")
except Exception as e:
    print(f"Could not load from HF path ({hf_path}): {e}")
    if os.path.exists(LOCAL_FALLBACK):
        print(f"Falling back to local file: {LOCAL_FALLBACK}")
        df = pd.read_csv(LOCAL_FALLBACK)
        print("Loaded dataset from local file.")
    else:
        print("No local fallback file found at", LOCAL_FALLBACK)
        print("Please ensure the dataset file exists either on Hugging Face or at the local fallback path.")
        sys.exit(1)

# === Data cleaning & preprocessing ===
print("Starting data cleaning...")
# Drop unnecessary columns if present
df.drop(columns=["CustomerID", "Unnamed: 0"], errors="ignore", inplace=True)

target_col = "ProdTaken"
if target_col not in df.columns:
    print(f"ERROR: target column '{target_col}' not found in dataframe columns: {df.columns.tolist()}")
    sys.exit(1)

# Drop rows where target is missing
df = df.dropna(subset=[target_col])

# Separate numeric and categorical columns
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# Fill numeric with median
for c in num_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].median())

# Fill categorical with mode
for c in cat_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].mode().iloc[0])

# Save cleaned dataset locally
os.makedirs(OUTPUT_DIR, exist_ok=True)
cleaned_path = os.path.join(OUTPUT_DIR, "tourism_cleaned.csv")
df.to_csv(cleaned_path, index=False)
print("Saved cleaned dataset to:", cleaned_path)

# === Train-test split ===
print("Splitting dataset into train and test...")
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()>1 else None)

# Write split files to current working directory (these will be uploaded)
Xtrain_path = "Xtrain.csv"
Xtest_path = "Xtest.csv"
ytrain_path = "ytrain.csv"
ytest_path = "ytest.csv"

Xtrain.to_csv(Xtrain_path, index=False)
Xtest.to_csv(Xtest_path, index=False)
ytrain.to_csv(ytrain_path, index=False)
ytest.to_csv(ytest_path, index=False)

print("Wrote split files:", [Xtrain_path, Xtest_path, ytrain_path, ytest_path])

# === Ensure dataset repo exists (create if missing) ===
try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Dataset repo '{REPO_ID}' exists. Will upload files to it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{REPO_ID}' not found. Attempting to create it...")
    try:
        create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False, token=HF_TOKEN)
        print("Created dataset repo:", REPO_ID)
    except HfHubHTTPError as e:
        print("Failed to create dataset repo:", e)
        sys.exit(1)
except Exception as e:
    print("Error while checking repo:", e)
    sys.exit(1)

# === Upload split files to HF dataset repo (file-by-file) ===
files_to_upload = [Xtrain_path, Xtest_path, ytrain_path, ytest_path]
for fp in files_to_upload:
    if not os.path.exists(fp):
        print(f"File not found (skipping): {fp}")
        continue
    print(f"Uploading '{fp}' to dataset repo '{REPO_ID}' ...")
    try:
        api.upload_file(
            path_or_fileobj=fp,
            path_in_repo=os.path.basename(fp),
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        print("Uploaded:", fp)
    except HfHubHTTPError as e:
        # helpful messages for common failures
        status = None
        try:
            status = e.response.status_code
        except Exception:
            pass
        print(f"Upload failed for {fp}: {e} (status={status})")
        if status == 401:
            print("Authorization error (401). Verify HF_TOKEN is valid and has 'datasets:write' or 'repo' scope.")
        elif status == 404:
            print("Repo not found (404). Check REPO_ID spelling.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected upload error for {fp}: {e}")
        sys.exit(1)

print("All done — split files uploaded to Hugging Face dataset:", REPO_ID)
