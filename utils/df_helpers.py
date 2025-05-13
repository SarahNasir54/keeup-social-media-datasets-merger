import os
import json
import pandas as pd
from glob import glob

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

def load_mediaeval15(folder_path):
    # Load tweet data
    tweet_cols = ['tweetId', 'tweetText', 'userId', 'imageId', 'username', 'timestamp', 'label']
    tweets_dev = pd.read_csv(os.path.join(folder_path, 'tweets_dev.txt'), sep='\t', names=tweet_cols, header=0)
    tweets_test = pd.read_csv(os.path.join(folder_path, 'tweets_test.txt'), sep='\t', names=tweet_cols, header=0)
    tweets = pd.concat([tweets_dev, tweets_test], ignore_index=True)

    # Load user features
    user_cols = ['tweetId', 'num_friends', 'num_followers', 'folfriend_ratio', 'times_listed', 'has_url', 'is_verified', 'num_tweets']
    user_dev = pd.read_csv(os.path.join(folder_path, 'user_features_dev.csv'), skipinitialspace=True)
    user_test = pd.read_csv(os.path.join(folder_path, 'user_features_test.txt'), names=user_cols, header=0)
    user_features = pd.concat([user_dev, user_test], ignore_index=True)

    # Load tweet-level features (e.g. retweets)
    tweet_feat_cols = ['tweetId', 'num_words', 'text_length', 'contains_questmark', 'num_questmark',
                       'contains_exclammark', 'num_exclammark', 'contains_happyemo', 'contains_sademo',
                       'contains_firstorderpron', 'contains_secondorderpron', 'contains_thirdorderpron',
                       'num_uppercasechars', 'num_possentiwords', 'num_negsentiwords', 'num_mentions',
                       'num_hashtags', 'num_URLs', 'num_retweets']
    
    tweet_feats_test = pd.read_csv(os.path.join(folder_path, 'tweet_features_test.txt'), names=tweet_feat_cols, header=0)
    tweet_feats_dev = pd.read_csv(os.path.join(folder_path, 'tweet_features_dev.csv'), skipinitialspace=True)
    tweet_features = pd.concat([tweet_feats_test, tweet_feats_dev], ignore_index=True)

    # Merge all
    df = tweets.merge(user_features, on='tweetId', how='left')
    df = df.merge(tweet_features[['tweetId', 'num_retweets']], on='tweetId', how='left')

    # Keep only required columns
    final_df = df[['tweetId', 'tweetText', 'timestamp', 'label', 'username',
                   'num_followers', 'num_friends', 'is_verified', 'num_retweets']]

    return final_df

def load_pheme5_original(main_folder):
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
                tweet_dir = os.path.join(label_path, tweet_folder, 'source-tweet')
                if not os.path.exists(tweet_dir):
                    continue

                # Load JSON file in the source-tweet folder
                for json_file in glob(os.path.join(tweet_dir, '*.json')):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            tweet = json.load(f)

                        user = tweet.get('user', {})
                        data.append({
                            'id': tweet.get('id'),
                            'text': tweet.get('text'),
                            'created_at': tweet.get('created_at'),
                            'label': label_type,
                            'followers_count': user.get('followers_count'),
                            'friends_count': user.get('friends_count'),
                            'verified': user.get('verified'),
                            'retweet_count': tweet.get('retweet_count'),
                            'favorite_count': tweet.get('favorite_count')
                        })
                    except Exception as e:
                        print(f"Error reading {json_file}: {e}")

    df = pd.DataFrame(data)
    return df

def load_pheme9(pheme_root):

    data = []
    threads_root = os.path.join(pheme_root, "threads")

    for lang in os.listdir(threads_root):  # 'en', 'de'
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

                try:
                    source_tweet_dir = os.path.join(tweet_path, "source-tweets")
                    annotation_file = os.path.join(tweet_path, "annotation.json")

                    tweet_files = os.listdir(source_tweet_dir)
                    if not tweet_files:
                        continue

                    tweet_file = os.path.join(source_tweet_dir, tweet_files[0])

                    with open(tweet_file, 'r', encoding='utf-8') as f:
                        tweet = json.load(f)

                    with open(annotation_file, 'r', encoding='utf-8') as f:
                        annotation = json.load(f)

                    data.append({
                        "id": tweet.get("id_str", tweet.get("id")),
                        "text": tweet.get("text"),
                        "created_at": tweet.get("created_at"),
                        "label": annotation.get("is_rumour", "unknown"),
                        "followers_count": tweet.get("user", {}).get("followers_count"),
                        "friends_count": tweet.get("user", {}).get("friends_count"),
                        "verified": tweet.get("user", {}).get("verified"),
                        "retweet_count": tweet.get("retweet_count"),
                        "favorite_count": tweet.get("favorite_count"),
                        "language": lang
                    })

                except Exception as e:
                    print(f"Error in {tweet_path}: {e}")

    return pd.DataFrame(data)

def load_phemeveracity(main_folder):
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
                tweet_dir = os.path.join(label_path, tweet_folder, 'source-tweets')
                if not os.path.exists(tweet_dir):
                    continue

                # Load JSON file in the source-tweet folder
                for json_file in glob(os.path.join(tweet_dir, '*.json')):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            tweet = json.load(f)

                        user = tweet.get('user', {})
                        data.append({
                            'id': tweet.get('id'),
                            'text': tweet.get('text'),
                            'created_at': tweet.get('created_at'),
                            'label': label_type,
                            'followers_count': user.get('followers_count'),
                            'friends_count': user.get('friends_count'),
                            'verified': user.get('verified'),
                            'retweet_count': tweet.get('retweet_count'),
                            'favorite_count': tweet.get('favorite_count')
                        })
                    except Exception as e:
                        print(f"Error reading {json_file}: {e}")

    df = pd.DataFrame(data)
    return df

def load_rumoureval17_dataset(root_path):
    data = []

    # Load label mappings from train and dev
    label_path = os.path.join(root_path, "traindev")
    with open(os.path.join(label_path, "rumoureval-subtaskB-train.json"), "r", encoding="utf-8") as f:
        train_labels = json.load(f)
    with open(os.path.join(label_path, "rumoureval-subtaskB-dev.json"), "r", encoding="utf-8") as f:
        dev_labels = json.load(f)

    all_labels = {**train_labels, **dev_labels}  # merge dictionaries

    threads_root = os.path.join(root_path, "rumoureval-data")

    for event in os.listdir(threads_root):
        event_path = os.path.join(threads_root, event)
        if not os.path.isdir(event_path):
            continue

        for tweet_folder in os.listdir(event_path):
            tweet_path = os.path.join(event_path, tweet_folder)
            if not os.path.isdir(tweet_path):
                continue

            try:
                source_tweet_dir = os.path.join(tweet_path, "source-tweet")
                tweet_files = os.listdir(source_tweet_dir)
                if not tweet_files:
                    continue

                tweet_file = os.path.join(source_tweet_dir, tweet_files[0])

                with open(tweet_file, 'r', encoding='utf-8') as f:
                    tweet = json.load(f)

                tweet_id = tweet.get("id_str", tweet.get("id"))
                label = all_labels.get(tweet_id, "unknown")

                data.append({
                    "id": tweet_id,
                    "text": tweet.get("text"),
                    "created_at": tweet.get("created_at"),
                    "label": label,
                    "name": tweet.get("user", {}).get("name"),
                    "user_followers_count": tweet.get("user", {}).get("followers_count"),
                    "user_friends_count": tweet.get("user", {}).get("friends_count"),
                    "user_verified": tweet.get("user", {}).get("verified"),
                    "retweet_count": tweet.get("retweet_count"),
                    "favorite_count": tweet.get("favorite_count")
                })

            except Exception as e:
                print(f"Error in {tweet_path}: {e}")

    return pd.DataFrame(data)

def load_rumoureval2019_dataset(root_path):
    data = []

    # Load label mappings
    with open(os.path.join(root_path, "train-key.json"), "r", encoding="utf-8") as f:
        train_labels = json.load(f).get("subtaskbenglish", {})

    with open(os.path.join(root_path, "dev-key.json"), "r", encoding="utf-8") as f:
        dev_labels = json.load(f).get("subtaskbenglish", {})

    all_labels = {**train_labels, **dev_labels}

    for dataset_folder in ["reddit-dev-data", "reddit-training-data", "twitter-english"]:
        dataset_path = os.path.join(root_path, dataset_folder)
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

                # Only look for the source tweet (not replies or nested paths)
                source_tweet_path = os.path.join(thread_path, "source-tweet")
                if not os.path.isdir(source_tweet_path):
                    continue

                try:
                    tweet_files = os.listdir(source_tweet_path)
                    if not tweet_files:
                        continue

                    # There should only be one file in source-tweet folder
                    tweet_file = os.path.join(source_tweet_path, tweet_files[0])
                    with open(tweet_file, "r", encoding="utf-8") as f:
                        tweet = json.load(f)

                    tweet_id = tweet.get("id_str", tweet.get("id"))
                    label = all_labels.get(tweet_id, "unknown")

                    data.append({
                        "post_id": tweet_id,
                        "post_text": tweet.get("text"),
                        "timestamp": tweet.get("created_at"),
                        "label": label,
                        "username": tweet.get("user", {}).get("name"),
                        "num_followers": tweet.get("user", {}).get("followers_count"),
                        "num_friends": tweet.get("user", {}).get("friends_count"),
                        "is_verified": tweet.get("user", {}).get("verified"),
                        "num_retweets": tweet.get("retweet_count")
                    })

                except Exception as e:
                    print(f"Error reading {source_tweet_path}: {e}")

    return pd.DataFrame(data)


def load_social_honeypot_dataset(main_folder):
    file_map = {
        'content_polluters_tweets.txt': 'polluter',
        'legitimate_users_tweets.txt': 'legitimate',
        'content_polluters.txt': 'polluter',
        'legitimate_users.txt': 'legitimate'
    }

    # Read user profiles (no headers)
    user_profiles = []
    for profile_file in ['content_polluters.txt', 'legitimate_users.txt']:
        path = os.path.join(main_folder, profile_file)
        df = pd.read_csv(path, sep='\t', header=None,
                         names=['UserID', 'ProfileCreatedAt', 'ProfileCollectedAt',
                                'NumberOfFollowings', 'NumberOfFollowers', 'NumberOfTweets',
                                'ScreenNameLength', 'DescriptionLength'])
        df['Label'] = file_map[profile_file]
        user_profiles.append(df)
    users_df = pd.concat(user_profiles, ignore_index=True)

    # Read tweets (no headers)
    tweet_data = []
    for tweet_file in ['content_polluters_tweets.txt', 'legitimate_users_tweets.txt']:
        path = os.path.join(main_folder, tweet_file)
        df = pd.read_csv(path, sep='\t', header=None,
                         names=['UserID', 'TweetID', 'TweetText', 'CreatedAt'])
        df['Label'] = file_map[tweet_file]
        tweet_data.append(df)
    tweets_df = pd.concat(tweet_data, ignore_index=True)

    # Merge tweets with user profiles on UserID and Label
    merged_df = tweets_df.merge(users_df, on=['UserID', 'Label'])

    # Select final columns
    final_df = merged_df[['TweetID', 'TweetText', 'CreatedAt', 'NumberOfFollowers', 'NumberOfFollowings', 'Label']]

    return final_df

def load_twitter(folder_path):
    # Define columns for each file
    post_cols_dev = ['post_id', 'post_text', 'user_id', 'image_id', 'username', 'timestamp', 'label']
    post_cols_test = ['post_id', 'post_text', 'user_id', 'username', 'image_id', 'timestamp']

    user_cols = ['post_id', 'num_friends', 'num_followers', 'folfriend_ratio', 'times_listed', 'has_url', 'is_verified', 'num_posts']
    post_feat_cols = ['post_id', 'num_words', 'text_length', 'contains_questmark', 'num_questmark',
                      'contains_exclammark', 'num_exclammark', 'contains_happyemo', 'contains_sademo',
                      'contains_firstorderpron', 'contains_secondorderpron', 'contains_thirdorderpron',
                      'num_uppercasechars', 'num_possentiwords', 'num_negsentiwords', 'num_mentions',
                      'num_hashtags', 'num_URLs', 'num_retweets']

    # Load dev set
    dev_path = os.path.join(folder_path, 'devset')
    posts_dev = pd.read_csv(os.path.join(dev_path, 'posts.txt'), sep='\t', names=post_cols_dev, header=0)
    users_dev = pd.read_csv(os.path.join(dev_path, 'user_features.txt'), sep=',', names=user_cols, header=0)
    feats_dev = pd.read_csv(os.path.join(dev_path, 'post_features.txt'), sep=',', names=post_feat_cols, header=0)

    # Merge dev
    dev = posts_dev.merge(users_dev, on='post_id', how='left')
    dev = dev.merge(feats_dev[['post_id', 'num_retweets']], on='post_id', how='left')

    # Load test set
    test_path = os.path.join(folder_path, 'testset')
    posts_test = pd.read_csv(os.path.join(test_path, 'posts.txt'), sep='\t', names=post_cols_test, header=0)
    posts_test["label"] = None  # No label in test set

    users_test = pd.read_csv(os.path.join(test_path, 'user_features.txt'), sep=',', names=user_cols, header=0)
    feats_test = pd.read_csv(os.path.join(test_path, 'post_features.txt'), sep=',', names=post_feat_cols, header=0)

    # Merge test
    test = posts_test.merge(users_test, on='post_id', how='left')
    test = test.merge(feats_test[['post_id', 'num_retweets']], on='post_id', how='left')

    # Concatenate dev and test
    df = pd.concat([dev, test], ignore_index=True)

    # Select only needed columns
    final_df = df[['post_id', 'post_text', 'timestamp', 'label', 'username',
                   'num_followers', 'num_friends', 'is_verified', 'num_retweets']]

    return final_df

def load_weibo_dataset(folder_path):
    # Files that include rumor/nonrumor data
    label_files = {
        'train_rumor.txt': 'rumor',
        'train_nonrumor.txt': 'nonrumor',
        'test_rumor.txt': 'rumor',
        'test_nonrumor.txt': 'nonrumor'
    }

    all_data = []

    for file_name, label in label_files.items():
        file_path = os.path.join(folder_path, file_name)

        if not os.path.exists(file_path):
            print(f"Warning: {file_name} not found in {folder_path}")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                continue  # Skip incomplete entries

            meta = lines[i].strip().split('|')
            tweet_text = lines[i + 2].strip()

            if len(meta) < 15:  # Basic sanity check
                continue

            all_data.append({
                'tweet_id': meta[0],
                'user_name': meta[1],
                'publish_time': meta[4],
                'user_auth_type': meta[10],
                'user_fans_count': meta[11],
                'user_follow_count': meta[12],
                'retweet_count': meta[6],
                'praise_count': meta[8],
                'tweet_content': tweet_text,
                'label': label
            })

    # Create base DataFrame
    df = pd.DataFrame(all_data)

    # Convert appropriate columns to numeric
    numeric_cols = ['user_fans_count', 'user_follow_count', 'retweet_count', 'praise_count']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    return df

def load_weibo_rumor_dataset(root_path):
    data = []

    txt_path = os.path.join(root_path, "Weibo.txt")
    json_dir = os.path.join(root_path, "Weibo")

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            event_id_part = parts[0]  # e.g., "eid:10031080900"
            label_part = parts[1]     # e.g., "label:0"

            event_id = event_id_part.replace("eid:", "")
            label = int(label_part.replace("label:", ""))

            json_path = os.path.join(json_dir, f"{event_id}.json")
            if not os.path.isfile(json_path):
                continue

            try:
                with open(json_path, 'r', encoding='utf-8') as jf:
                    posts = json.load(jf)

                # Find the post where id == event_id
                root_post = next((post for post in posts if str(post.get("id")) == event_id), None)
                if not root_post:
                    continue

                data.append({
                    "id": root_post.get("id"),
                    "original_text": root_post.get("original_text", ""),
                    "username": root_post.get("username", ""),
                    "followers_count": root_post.get("followers_count", 0),
                    "friends_count": root_post.get("friends_count", 0),
                    "verified": root_post.get("verified", False),
                    "reposts_count": root_post.get("reposts_count", 0),
                    "favourites_count": root_post.get("favourites_count", 0),
                    "label": label
                })

            except Exception as e:
                print(f"Error reading {json_path}: {e}")

    return pd.DataFrame(data)