import os
import argparse
import json
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from tqdm import tqdm

#======================================================================#
def load_split(split_dir):
    """Load one split (train/test) into a Hugging Face Dataset."""
    samples = []
    for fname in tqdm(os.listdir(split_dir), ncols=100, desc=f"Loading {split_dir}"):
        if not fname.endswith(".npz"):
            continue
        path = os.path.join(split_dir, fname)
        data = np.load(path, allow_pickle=True)

        sample = {"case_id": fname}
        for key, value in data.items():
            if key == "_metadata":
                # metadata is stored as JSON string inside npz
                sample["metadata"] = json.loads(value[0])["metadata"]
            else:
                sample[key] = value.astype(np.float32) if value.dtype == np.float32 else value.astype(np.int64)

        samples.append(sample)

    return Dataset.from_list(samples)

def build_dataset(basedir: str):
    """Load train/test splits into a DatasetDict."""
    train_dir = os.path.join(basedir, "train")
    test_dir = os.path.join(basedir, "test")

    train_ds = load_split(train_dir)
    test_ds = load_split(test_dir)

    return DatasetDict({"train": train_ds, "test": test_ds})

#======================================================================#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LPBF dataset for Hugging Face.")
    parser.add_argument("--basedir", type=str, required=True,
                        help="Base directory containing LPBF/train and LPBF/test")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="Hugging Face repo id to push to")
    parser.add_argument("--test", action="store_true",
                        help="Test uploaded dataset")
    args = parser.parse_args()

    if args.test:
        print(f"Testing mode. Evaluating dataset at {args.repo_id}.")
        from datasets import load_dataset
        dataset = load_dataset(args.repo_id)
        print(dataset)
        exit()

    dataset = build_dataset(args.basedir)

    print(dataset)

    # Save locally
    dataset.save_to_disk("LPBF_hf")
    print("Saved dataset to LPBF_hf/")
    
    # Ask for confirmation before pushing to Hugging Face Hub
    confirmation = input(f"Push to {args.repo_id}? (y/N): ")
    if confirmation.lower() not in ["y", "yes"]:
        print("Push cancelled.")
    else:
        dataset.push_to_hub(args.repo_id)
        print(f"Pushed dataset to {args.repo_id}")

    exit()
