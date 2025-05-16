import os
import json
import pandas as pd
from glob import glob

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


def load_pheme5_reposts(path):
    repost_data = []

    for event_name in os.listdir(path):
        event_path = os.path.join(path, event_name)
        if not os.path.isdir(event_path):
            continue

        for label_type in ['rumours', 'non-rumours']:
            label_path = os.path.join(event_path, label_type)
            if not os.path.isdir(label_path):
                continue

            for tweet_folder in os.listdir(label_path):
                reactions_dir = os.path.join(label_path, tweet_folder, 'reactions')
                if not os.path.exists(reactions_dir):
                    continue

                for json_file in glob(os.path.join(reactions_dir, '*.json')):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            tweet = json.load(f)

                        original_id = tweet.get('in_reply_to_status_id')
                        if original_id is None:
                            continue  # skip if it's not a reply

                        repost_data.append({
                            'id': str(original_id),          # original tweet ID
                            'text': tweet.get('text', ''),
                            'label': label_type
                        })
                    except Exception as e:
                        print(f"Error reading {json_file}: {e}")

    return pd.DataFrame(repost_data)


def load_pheme9_reposts(path):
    data = []
    threads_root = os.path.join(path, "threads")

    for lang in os.listdir(threads_root):  # 'en', 'de', etc.
        lang_dir = os.path.join(threads_root, lang)
        if not os.path.isdir(lang_dir):
            continue

        for event in os.listdir(lang_dir):
            event_path = os.path.join(lang_dir, event)
            if not os.path.isdir(event_path):
                continue

            for tweet_folder in os.listdir(event_path):
                tweet_path = os.path.join(event_path, tweet_folder)
                if not os.path.isdir(tweet_path):
                    continue

                reactions_dir = os.path.join(tweet_path, "reactions")
                annotation_file = os.path.join(tweet_path, "annotation.json")

                try:
                    with open(annotation_file, 'r', encoding='utf-8') as f:
                        annotation = json.load(f)
                    label = annotation.get("is_rumour", "unknown")
                except Exception as e:
                    print(f"Error reading annotation in {tweet_path}: {e}")
                    label = "unknown"

                if not os.path.isdir(reactions_dir):
                    continue

                for file in os.listdir(reactions_dir):
                    if not file.endswith('.json'):
                        continue

                    file_path = os.path.join(reactions_dir, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            tweet = json.load(f)

                        parent_id = tweet.get("in_reply_to_status_id")
                        if parent_id is None:
                            continue

                        data.append({
                            "id": str(parent_id),  # original tweet ID
                            "text": tweet.get("text", ""),
                            "label": label,
                        })

                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    return pd.DataFrame(data)

def load_phemeveracity_reposts(main_folder):
    data = []

    # Traverse event folders
    for event_name in os.listdir(main_folder):
        event_path = os.path.join(main_folder, event_name)
        if not os.path.isdir(event_path):
            continue

        for label_type in ['rumours', 'non-rumours']:
            label_path = os.path.join(event_path, label_type)
            if not os.path.isdir(label_path):
                continue

            for tweet_folder in os.listdir(label_path):
                tweet_dir = os.path.join(label_path, tweet_folder)
                reactions_dir = os.path.join(tweet_dir, 'reactions')

                if not os.path.exists(reactions_dir):
                    continue

                for json_file in glob(os.path.join(reactions_dir, '*.json')):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            tweet = json.load(f)

                        parent_id = tweet.get("in_reply_to_status_id")
                        if parent_id is None:
                            continue

                        data.append({
                            'id': str(parent_id),  # original tweet ID
                            'text': tweet.get('text', ''),
                            'label': label_type
                        })

                    except Exception as e:
                        print(f"Error reading {json_file}: {e}")

    return pd.DataFrame(data)

def load_rumoureval17_reposts(path):
    data = []

    # Load label mappings
    label_path = os.path.join(path, "traindev")
    with open(os.path.join(label_path, "rumoureval-subtaskB-train.json"), "r", encoding="utf-8") as f:
        train_labels = json.load(f)
    with open(os.path.join(label_path, "rumoureval-subtaskB-dev.json"), "r", encoding="utf-8") as f:
        dev_labels = json.load(f)

    all_labels = {**train_labels, **dev_labels}

    threads_root = os.path.join(path, "rumoureval-data")

    for event in os.listdir(threads_root):
        event_path = os.path.join(threads_root, event)
        if not os.path.isdir(event_path):
            continue

        for tweet_folder in os.listdir(event_path):
            tweet_path = os.path.join(event_path, tweet_folder)
            if not os.path.isdir(tweet_path):
                continue

            reactions_dir = os.path.join(tweet_path, "replies")
            if not os.path.isdir(reactions_dir):
                continue

            for file in os.listdir(reactions_dir):
                if not file.endswith('.json'):
                    continue

                file_path = os.path.join(reactions_dir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        tweet = json.load(f)

                    parent_id = tweet.get("in_reply_to_status_id")
                    if parent_id is None:
                        continue

                    label = all_labels.get(str(parent_id), "unknown")

                    data.append({
                        "id": str(parent_id),
                        "text": tweet.get("text", ""),
                        "label": label
                    })

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return pd.DataFrame(data)

def load_rumoureval2019_reposts(path):
    data = []

    # Load labels from train and dev keys
    with open(os.path.join(path, "train-key.json"), "r", encoding="utf-8") as f:
        train_labels = json.load(f).get("subtaskbenglish", {})

    with open(os.path.join(path, "dev-key.json"), "r", encoding="utf-8") as f:
        dev_labels = json.load(f).get("subtaskbenglish", {})

    all_labels = {**train_labels, **dev_labels}

    for dataset_folder in ["reddit-dev-data", "reddit-training-data", "twitter-english"]:
        dataset_path = os.path.join(path, dataset_folder)
        if not os.path.isdir(dataset_path):
            continue

        for event in os.listdir(dataset_path):
            event_path = os.path.join(dataset_path, event)
            if not os.path.isdir(event_path):
                continue

            for thread_folder in os.listdir(event_path):
                thread_path = os.path.join(event_path, thread_folder)
                if not os.path.isdir(thread_path):
                    continue

                replies_path = os.path.join(thread_path, "replies")
                if not os.path.isdir(replies_path):
                    continue

                for reply_file in os.listdir(replies_path):
                    if not reply_file.endswith('.json'):
                        continue

                    reply_path = os.path.join(replies_path, reply_file)
                    try:
                        with open(reply_path, "r", encoding="utf-8") as f:
                            tweet = json.load(f)

                        parent_id = tweet.get("in_reply_to_status_id_str") or tweet.get("in_reply_to_status_id")
                        if parent_id is None:
                            continue

                        label = all_labels.get(str(parent_id), "unknown")

                        data.append({
                            "id": str(parent_id),
                            "text": tweet.get("text", ""),
                            "label": label
                        })

                    except Exception as e:
                        print(f"Error reading {reply_path}: {e}")

    return pd.DataFrame(data)

def load_weibo_rumor_reposts(path):
    data = []

    txt_path = os.path.join(path, "Weibo.txt")
    json_dir = os.path.join(path, "Weibo")

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            event_id_part = parts[0]  
            label_part = parts[1]     
            event_id = event_id_part.replace("eid:", "")
            label = int(label_part.replace("label:", ""))

            json_path = os.path.join(json_dir, f"{event_id}.json")
            if not os.path.isfile(json_path):
                continue

            try:
                with open(json_path, 'r', encoding='utf-8') as jf:
                    posts = json.load(jf)

                for post in posts:
                    # Skip the root post (id == event_id)
                    if str(post.get("id")) == event_id:
                        continue

                    data.append({
                        "id": post.get("id"),
                        "text": post.get("text", ""),
                        "label": label
                    })

            except Exception as e:
                print(f"Error reading {json_path}: {e}")

    return pd.DataFrame(data)