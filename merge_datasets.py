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
    "CED" : [r"D:\text datasets\text datasets\CED_Dataset", "zh", "society,disasters,health,politics,science", "Weibo"],
    #"FbMultiLingMisinfo": [r'D:\text datasets\text datasets\FbMultiLingMisinfo.csv', "en,it,de,es,fr,pt,id,nl,tl,el,af,hr,da", "politics", "Facebook"],
    #"MediaEval15": [r"D:\text+image\text+image\mediaeval2015", "en,es,id,fr,no,pt,it,nl,ar", "society,disasters,health,politics,science,technology,entertainment,publicsafety", "Twitter"],
    #"Pheme5": [r'D:\text datasets\text datasets\phemernrdataset\pheme-rnr-dataset', "en","disaster,crime,publicsafety,religion", "Twitter"],
    #"Pheme9": [r'D:\text+image\text+image\pheme9\pheme-rumour-scheme-dataset', "en,de", "disaster,crime,publicsafety,religion", "Twitter"],
    #"Pheme-veracity": [r'D:\text datasets\text datasets\PHEME_veracity\all-rnr-annotated-threads', "en", "disaster,crime,publicsafety,religion", "Twitter"],
    #"RumorEval17": [r"D:\text datasets\text datasets\RumorEval17\semeval2017-task8-dataset", "en", "others", "Twitter"],
    #"RumorEval19": [r"D:\text datasets\text datasets\rumoureval2019\rumoureval2019\rumoureval-2019-training-data\rumoureval-2019-training-data", "en", "others", "Twitter,Reddit"],
    #"Social-Honeypot": [r'D:\text datasets\text datasets\Social-Honeypot\social_honeypot_icwsm_2011', "en,es,ms,pt", "spam,politics,jobs,social", "Twitter"],
    #"Twitter": [r'D:\text+image\text+image\twitter', "en,es,id,fr,no,pt,it,nl,ar", "society,disasters,health,politics,science,technology,entertainment,publicsafety", "Twitter"],
    #"Weibo-data": [r'D:\text+image\text+image\Weibo-dataset-main\Weibo-dataset-main', "zh", "society,disasters,health,politics,science,technology,entertainment,publicsafety", "Weibo"],
    "Weibo-Rumor": [r"D:\text datasets\text datasets\weibo rumor", "zh", "politics", "Weibo"],
}

# fill this dictionary for only special cases
convert_to_df = {
    "CED" : load_ced_original_posts,
    "Social-Honeypot": load_social_honeypot_dataset,
    "Weibo-data": load_weibo_dataset,
    "MediaEval15": load_mediaeval15,
    "Pheme5": load_pheme5_original,
    "Pheme9": load_pheme9,
    "Pheme-veracity": load_phemeveracity,
    "RumorEval17": load_rumoureval17_dataset,
    "RumorEval19": load_rumoureval2019_dataset,
    "Twitter": load_twitter, 
    "Weibo-Rumor": load_weibo_rumor_dataset
}

convert_to_reposts = {
    "CED": load_ced_repost_posts,
}

def update_fields(dataset_name, df):
    required_columns = [
        "post_id", "text", "timestamp", "label", "username",
        "follower_count", "friends_count", "is_verified",
        "repost_text", "repost_count", "likes", "language", "domain", "platform"
    ]
    
    # Check for missing columns and add them with default values
    for col in required_columns:
        if col not in df.columns:
            # Set default values based on the column type
            if col in ["username", "text", "repost_text", "timestamp", "is_verified"]:
                df[col] = ""
            else:
                df[col] = 0
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
        df = df[df['label'].isin(labels_to_keep)]
        df['label'] = df['label'].astype(str).str.strip().map(label_mapping)

    # Clean text fields
    df["text"] = df["text"].astype(str).apply(clean_text)
    #df["repost_text"] = df["repost_text"].astype(str).apply(clean_text)

    df["is_verified"] = df["is_verified"].astype(bool)

    df["post_id"] = df["post_id"].astype(str).str.strip()

    # Timestamp parsing
    df["timestamp"] = df["timestamp"].apply(standardize_timestamp)

    df["language"] = DATA_PATHS[dataset_name][1]
    df["domain"] = DATA_PATHS[dataset_name][2]
    df["platform"] = DATA_PATHS[dataset_name][3]


    return df[["post_id", "text", "timestamp", "label", "username", "follower_count", "friends_count", "is_verified", "repost_count", "likes", "language", "domain", "platform"]] #, df[["post_id", "repost_text", "label"]]



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
        futures = {
            executor.submit(process_dataset, name, DATA_PATHS[name][0]): name
            for name in DATA_PATHS
        }
        
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
