import os
import json
import pandas as pd

def load_ced_original_posts(path):

    original_path = os.path.join(path, 'original-microblog')
    repost_paths = {
        'rumor': os.path.join(path, 'rumor-repost'),
        'nonrumor': os.path.join(path, 'non-rumor-repost')
    }

    def load_json(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    original_records = []

    for filename in os.listdir(original_path):
        if not filename.endswith('.json'):
            continue

        microblog_id = filename.replace('.json', '')
        original_file = os.path.join(original_path, filename)

        try:
            data = load_json(original_file)

            user = data.get("user", {})
            if not isinstance(user, dict):
                user = {}

            label = None
            for lbl, path in repost_paths.items():
                if os.path.exists(os.path.join(path, filename)):
                    label = lbl
                    break
            if label is None:
                continue

            original_records.append({
                "id": microblog_id,
                "text": data.get("text", ""),
                "time": data.get("time", None),
                "followers": user.get("followers", None),
                "friends": user.get("friends", None),
                "verified": user.get("verified", False),
                "reposts": data.get("reposts", 0),
                "likes": data.get("likes", 0),
                "label": label,
            })

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return pd.DataFrame(original_records)
