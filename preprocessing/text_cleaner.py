import re
import string
import pandas as pd
from datetime import datetime

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # normalize whitespace
    return text

def standardize_timestamp(value):
 # Try UNIX timestamp
    try:
        return pd.to_datetime(int(float(value)), unit='s')
    except:
        pass

    # Try Twitter-style string
    try:
        return pd.to_datetime(value, format='%a %b %d %H:%M:%S %z %Y')
    except:
        pass

    # Try ISO-style or general datetime string (e.g. "2012-11-13 16:55")
    try:
        return pd.to_datetime(value, errors='coerce')
    except:
        return pd.NaT


