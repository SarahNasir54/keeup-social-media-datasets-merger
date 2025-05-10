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
    try:
        # If it's numeric, convert from UNIX timestamp (int)
        return pd.to_datetime(int(value), unit='s')
    except (ValueError, TypeError):
        pass  # Ignore and continue to datetime string conversion

    # If the value is not numeric, try to convert it as a datetime string
    try:
        return pd.to_datetime(value, format='%a %b %d %H:%M:%S %z %Y', errors='coerce')
    except (ValueError, TypeError):
        return pd.NaT


