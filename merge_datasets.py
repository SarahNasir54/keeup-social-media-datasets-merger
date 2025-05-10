import os
import pandas as pd
from preprocessing.text_cleaner import clean_text, standardize_timestamp
from utils.df_helpers import *
from utils.repost_helpers import *
from utils.io_helpers import load_dataset, save_dataset, load_mappings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import traceback

DATA_PATHS = {
    "CED" : r"E:\social media datasets\CED\Chinese_Rumor_Dataset-master\CED_Dataset",
    "FbMultiLingMisinfo": r'E:\social media datasets\Fbmultilingual.csv'
}

# fill this dictionary for only special cases
convert_to_df = {
    "CED" : load_ced_original_posts,
}

convert_to_reposts = {
    "CED": load_ced_repost_posts,
}

def update_fields(dataset_name, df):
    required_columns = [
        "post_id", "text", "timestamp", "label", "username",
        "follower_count", "friends_count", "is_verified",
        "repost_text", "repost_count", "likes"
    ]
    
    # Check for missing columns and add them with default values
    for col in required_columns:
        if col not in df.columns:
            # Set default values based on the column type
            if col in ["username", "text", "repost_text", "timestamp"]:
                df[col] = ""
            else:
                df[col] = 0
            # Special case for boolean columns
            if col == "is_verified":
                df[col] = False
            if col == "label":
                df[col] = "unknown"

    return df


def process_and_map(dataset_name, df, mapping):
    column_mapping = mapping.get("column_mapping", {})
    label_mapping = mapping.get("label_mapping", {})
    repost_mapping = mapping.get("repost_mapping", {})

    df = update_fields(dataset_name, df)

    # Rename columns to standard names
    df = df[[column_mapping["post_id"], column_mapping["text"], column_mapping["timestamp"], column_mapping["label"], column_mapping["username"], column_mapping["follower_count"], column_mapping["friends_count"], column_mapping["is_verified"], column_mapping["repost_count"], column_mapping["likes"]]].dropna()
    
    df = df.rename(columns={
        column_mapping["post_id"]: "post_id",
        column_mapping["text"]: "text",
        column_mapping["timestamp"]: "timestamp",
        column_mapping["label"]: "label",
        column_mapping["username"]: "username",
        column_mapping["follower_count"]: "follower_count",
        column_mapping["friends_count"]: "friends_count",
        column_mapping["is_verified"]: "is_verified",
        column_mapping["repost_count"]: "repost_count",
        column_mapping["likes"]: "likes"
    })

    # df = df[[repost_mapping["post_id"], repost_mapping["repost_text"], repost_mapping["label"]]]

    # df = df.rename(columns={
    #     repost_mapping["post_id"]: "post_id",
    #     repost_mapping["repost_text"]: "repost_text",
    #     repost_mapping["label"]: "label",
    # })

    # Map labels to standard labels
    labels_to_keep = list(label_mapping.keys())
    
    if len(labels_to_keep) != 0:

        # Convert the labels to string
        df['label'] = df['label'].astype(str)

        # Only select the rows which contain the specific labels defined in the mapping values
        df = df[df['label'].isin(labels_to_keep)]

        # Map credible to real and not credible to fake
        df['label'] = df['label'].astype(str).str.strip().map(label_mapping)


    # Clean text fields
    df["text"] = df["text"].astype(str).apply(clean_text)
    #df["repost_text"] = df["repost_text"].astype(str).apply(clean_text)

    # Timestamp parsing
    df["timestamp"] = df["timestamp"].apply(standardize_timestamp)


    return df[["post_id", "text", "timestamp", "label", "username", "follower_count", "friends_count", "is_verified", "repost_count", "likes"]] #, df[["post_id", "repost_text", "label"]]



def main():
    mappings = load_mappings("configs/mappings.json")
    originals_list = []
    reposts_list = []


    def process_dataset(dataset_name, path):
  
        if dataset_name in convert_to_df.keys():
            df = convert_to_df[dataset_name](path)
        else:
            df = load_dataset(path)  
        return process_and_map(dataset_name, df, mappings[dataset_name])

    # Use ThreadPoolExecutor to process datasets concurrently
    with ThreadPoolExecutor(max_workers=25) as executor:
        futures = {executor.submit(process_dataset, name, path): name for name, path in DATA_PATHS.items()}
        
        # Show progress with tqdm
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing datasets"):
            try:
                originals = future.result()
                originals_list.append(originals)
                #reposts_list.append(reposts)
            except Exception as e:
                name = futures[future]
                print(f"Error processing {name}: {e}")
                traceback.print_exc()

    # Save the final merged datasets
    os.makedirs("processed_data", exist_ok=True)
    all_originals = pd.concat(originals_list, ignore_index=True)
    #all_reposts = pd.concat(reposts_list, ignore_index=True)
    save_dataset(all_originals, "processed_data/all_originals.parquet")
    #save_dataset(all_reposts, "processed_data/all_reposts.parquet") | Reposts: {len(all_reposts)}
    print(f"Merged Social Media dataset saved with {len(all_originals)} records.") 

# Run the main function if this script is executed
if __name__ == '__main__':
    main()
