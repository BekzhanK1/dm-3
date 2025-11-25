import pickle
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

VOCAB_SIZE = 10_000
OOV_TOKEN = "<OOV>"
MAX_LENGTH = 500  # Updated to 500 as per outline
DEFAULT_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
TOKEN_FILTERS = DEFAULT_FILTERS.replace("!", "").replace("?", "")

DATA_DIR = Path(__file__).resolve().parent
FAKE_PATH = DATA_DIR / "Fake.csv"
TRUE_PATH = DATA_DIR / "True.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.pkl"


def clean_text_lstm(text: str) -> str:
    """Preprocess news text specifically for LSTM usage."""
    if not isinstance(text, str):
        return ""

    # Remove typical attribution patterns such as "WASHINGTON (Reuters) -"
    text = re.sub(
        r"\b[A-Z][A-Z\s]+\s*\(reuters\)\s*-\s*",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    # Drop the standalone word "reuters" anywhere in the text
    text = re.sub(r"reuters", " ", text, flags=re.IGNORECASE)
    # Remove URLs and common social media artifacts
    text = re.sub(r"pic\.twitter\.com/\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"http\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"@\w+", " ", text)

    text = text.lower()
    # Keep alphanumeric characters along with ! and ?
    text = re.sub(r"[^a-z0-9!?]+", " ", text)
    # Ensure ! and ? are tokenized separately to preserve emphasis
    text = re.sub(r"([!?])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_prepare_dataframe(fake_path: Path, true_path: Path) -> pd.DataFrame:
    fake_df = pd.read_csv(fake_path)
    fake_df["label"] = 0  # Fake = 0 as per outline request "label: 0=Fake, 1=True" -> Wait, outline says "label: 0=Fake, 1=True"
    # Actually, usually Fake is 1 in detection tasks, but I will strictly follow the outline:
    # "label: 0=Fake, 1=True"
    # Wait, let me double check the outline.
    # "label: 0=Fake, 1=True"
    
    fake_df["label"] = 0
    
    true_df = pd.read_csv(true_path)
    true_df["label"] = 1

    for df in (fake_df, true_df):
        df["title"] = df["title"].fillna("")
        df["text"] = df["text"].fillna("")
        df["combined_text"] = (df["title"] + " " + df["text"]).str.strip()

    combined_df = (
        pd.concat([fake_df, true_df], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    combined_df["clean_text"] = combined_df["combined_text"].apply(clean_text_lstm)
    return combined_df


def tokenize_and_pad(texts: pd.Series) -> Tuple[np.ndarray, Tokenizer]:
    tokenizer = Tokenizer(
        num_words=VOCAB_SIZE,
        oov_token=OOV_TOKEN,
        filters=TOKEN_FILTERS,
        lower=False,
        split=" ",
    )
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(
        sequences, maxlen=MAX_LENGTH, padding="post", truncating="post"
    )
    return padded, tokenizer


def get_data(subset_fraction: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tokenizer]:
    """
    Loads data, preprocesses, splits into Train (70%), Val (15%), Test (15%).
    Returns: X_train, y_train, X_val, y_val, X_test, y_test, tokenizer
    
    Args:
        subset_fraction: Fraction of data to use (e.g., 0.3 for 30%). 
                         If < 1.0, it samples BEFORE splitting.
    """
    # Note: We always load from scratch if subsetting to ensure correct sampling
    # Or we can load full and sample. Let's load full and sample.
    
    if PROCESSED_DATA_PATH.exists() and subset_fraction == 1.0:
        print(f"Loading processed data from {PROCESSED_DATA_PATH}...")
        with open(PROCESSED_DATA_PATH, "rb") as f:
            return pickle.load(f)

    print(f"Processing data from scratch (Subset: {subset_fraction*100}%)...")
    print(f"  Loading raw data from {FAKE_PATH} and {TRUE_PATH}...")
    combined_df = load_and_prepare_dataframe(FAKE_PATH, TRUE_PATH)
    
    if subset_fraction < 1.0:
        print(f"  Subsampling {subset_fraction*100}% of data...")
        combined_df = combined_df.sample(frac=subset_fraction, random_state=42).reset_index(drop=True)
        
    print(f"  Data loaded. Total samples: {len(combined_df)}")
    
    print("  Tokenizing and padding text data...")
    padded_sequences, tokenizer = tokenize_and_pad(combined_df["clean_text"])
    labels = combined_df["label"].to_numpy(dtype=np.int32)
    print(f"  Tokenization complete. Shape: {padded_sequences.shape}")

    print("  Splitting data into Train (70%), Val (15%), Test (15%)...")
    # First split: 70% Train, 30% Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        padded_sequences,
        labels,
        test_size=0.3,
        random_state=42,
        stratify=labels,
    )

    # Second split: Split Temp into 50% Val, 50% Test (which is 15% of total each)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )
    
    print("  Data splitting complete.")
    data = (X_train, y_train, X_val, y_val, X_test, y_test, tokenizer)
    
    if subset_fraction == 1.0:
        print(f"  Saving processed data to {PROCESSED_DATA_PATH}...")
        with open(PROCESSED_DATA_PATH, "wb") as f:
            pickle.dump(data, f)
        print("  Data saved successfully.")
        
    return data


def main():
    X_train, y_train, X_val, y_val, X_test, y_test, tokenizer = get_data()
    
    print(f"Tokenizer vocabulary size: {VOCAB_SIZE}")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)


if __name__ == "__main__":
    main()

