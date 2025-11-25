import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


VOCAB_SIZE = 10_000
OOV_TOKEN = "<OOV>"
MAX_LENGTH = 256
DEFAULT_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
TOKEN_FILTERS = DEFAULT_FILTERS.replace("!", "").replace("?", "")

DATA_DIR = Path(__file__).resolve().parent
FAKE_PATH = DATA_DIR / "Fake.csv"
TRUE_PATH = DATA_DIR / "True.csv"


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
    fake_df["label"] = 1

    true_df = pd.read_csv(true_path)
    true_df["label"] = 0

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
    return combined_df, true_df


def tokenize_and_pad(texts: pd.Series):
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


def display_sample_reuters_removal(true_df: pd.DataFrame) -> None:
    mask = true_df["combined_text"].str.contains("reuters", case=False, na=False)
    if mask.any():
        sample_raw = true_df.loc[mask, "combined_text"].iloc[0]
    else:
        sample_raw = true_df["combined_text"].iloc[0]

    sample_clean = clean_text_lstm(sample_raw)
    print("Raw True sample:\n", sample_raw[:400], "\n")
    print("Cleaned True sample (Reuters removed):\n", sample_clean[:400], "\n")


def main():
    combined_df, true_df = load_and_prepare_dataframe(FAKE_PATH, TRUE_PATH)

    padded_sequences, tokenizer = tokenize_and_pad(combined_df["clean_text"])
    labels = combined_df["label"].to_numpy(dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    print(f"Tokenizer vocabulary size (capped): {VOCAB_SIZE}")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    display_sample_reuters_removal(true_df)


if __name__ == "__main__":
    main()

