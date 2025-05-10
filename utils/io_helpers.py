import pandas as pd
import json
import os

def load_dataset(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext == ".json":
        return pd.read_json(file_path)
    elif ext == ".txt":
        # Try to load as a tab-separated or comma-separated file
        try:
            return pd.read_csv(file_path, sep="\t")  # Try TSV
        except Exception:
            return pd.read_csv(file_path, sep=",", header=None)  # Fallback to CSV without headers
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def save_dataset(df, path):
    df.to_parquet(path, index=False)

def load_mappings(mapping_file):
    with open(mapping_file, "r", encoding="utf-8") as f:
        return json.load(f)
