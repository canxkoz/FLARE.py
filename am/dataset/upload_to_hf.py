import os
import argparse
import json
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo
from tqdm import tqdm

#======================================================================#
def build_split(split_dir, split_name):
    """Scan one split (train/test) and build metadata table with file references."""
    samples = []
    for fname in tqdm(sorted(os.listdir(split_dir)), ncols=100, desc=f"Indexing {split_name}"):
        if not fname.endswith(".npz"):
            continue
        path = os.path.join(split_dir, fname)
        data = np.load(path, allow_pickle=True)

        # extract simple metadata
        num_nodes = data["pos"].shape[0]
        num_edges = data["edge_index"].shape[1] if "edge_index" in data else 0

        sample = {
            "case_id": fname.replace(".npz", ""),
            "file": os.path.join(split_name, fname),  # relative path inside repo
            "num_nodes": int(num_nodes),
            "num_edges": int(num_edges),
        }
        if "_metadata" in data:
            sample["metadata"] = json.dumps(json.loads(data["_metadata"][0])["metadata"])
        else:
            sample["metadata"] = "{}"

        samples.append(sample)

    return Dataset.from_pandas(pd.DataFrame(samples))

def build_dataset(basedir: str):
    """Load train/test splits into a DatasetDict of file references."""
    train_ds = build_split(os.path.join(basedir, "train"), "train")
    test_ds = build_split(os.path.join(basedir, "test"), "test")
    return DatasetDict({"train": train_ds, "test": test_ds})

def upload_npz_files(basedir: str, repo_id: str):
    """Upload all NPZ files to the HuggingFace repository using upload_folder."""
    api = HfApi()

    # Create repository if it doesn't exist
    try:
        create_repo(repo_id, exist_ok=True)
        print(f"Repository {repo_id} ready")
    except Exception as e:
        print(f"Repository setup: {e}")

    # Upload entire LPBF folder structure in one efficient operation
    print(f"Uploading NPZ files from {basedir}...")
    api.upload_folder(
        folder_path=basedir,
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns="**/*.npz",  # Only upload NPZ files
        commit_message="Upload LPBF dataset NPZ files"
    )

    print("All NPZ files uploaded successfully!")

#======================================================================#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LPBF dataset for Hugging Face.")
    parser.add_argument("--basedir", type=str, required=True,
                        help="Base directory containing LPBF/train and LPBF/test")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="Hugging Face repo id to push to")
    parser.add_argument("--upload-files", action="store_true",
                        help="Upload NPZ files to HuggingFace repository")
    args = parser.parse_args()

    # Build dataset index
    dataset = build_dataset(args.basedir)
    print(dataset)

    # Save locally
    dataset.save_to_disk("LPBF_hf")
    print("Saved dataset to LPBF_hf/")

    if args.upload_files:
        print(f"\nUploading NPZ files to {args.repo_id}...")
        upload_npz_files(args.basedir, args.repo_id)
        
        print(f"\nUploading dataset index to {args.repo_id}...")
        dataset.push_to_hub(args.repo_id)
        print(f"Upload complete! Repository {args.repo_id} now contains:")
        print("  - NPZ files: train/*.npz, test/*.npz")
        print("  - Dataset index: parquet files referencing the NPZ files")
    else:
        confirmation = input(f"Upload dataset index to {args.repo_id}? (y/N): ")
        if confirmation.lower() in ["y", "yes"]:
            dataset.push_to_hub(args.repo_id)
            print(f"Pushed dataset index to {args.repo_id}")
            print("Note: Use --upload-files flag to also upload NPZ files")
        else:
            print("Upload cancelled.")
            print("To upload NPZ files and dataset, use:")
            print(f"  python {__file__} --basedir {args.basedir} --repo_id {args.repo_id} --upload-files")
