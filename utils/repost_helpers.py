import os
import json
import pandas as pd

def load_ced_repost_posts(path):

    original_path = os.path.join(path, 'original-microblog')
    repost_paths = {
        'rumor': os.path.join(path, 'rumor-repost'),
        'nonrumor': os.path.join(path, 'non-rumor-repost')
    }

    def load_json(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    repost_records = []

    for filename in os.listdir(original_path):
        if not filename.endswith('.json'):
            continue

        microblog_id = filename.replace('.json', '')
        repost_file = None
        label = None

        for lbl, path in repost_paths.items():
            candidate = os.path.join(path, filename)
            if os.path.exists(candidate):
                repost_file = candidate
                label = lbl
                break

        if repost_file is None:
            continue

        try:
            repost_data = load_json(repost_file)
            for repost in repost_data:
                repost_records.append({
                    "id": microblog_id,
                    "text": repost.get("text", ""),
                    "label": label
                })

        except Exception as e:
            print(f"Error processing repost file {filename}: {e}")

    return pd.DataFrame(repost_records)
