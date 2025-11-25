import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


VOCAB_SIZE = 10_000
MAX_LEN = 256
OOV_TOKEN = "<OOV>"
DEFAULT_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
TOKEN_FILTERS = DEFAULT_FILTERS.replace("!", "").replace("?", "")

DATA_DIR = Path(__file__).resolve().parent
FAKE_PATH = DATA_DIR / "Fake.csv"
TRUE_PATH = DATA_DIR / "True.csv"

X_TRAIN = None
X_VAL = None
Y_TRAIN = None
Y_VAL = None


def clean_text_lstm(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(
        r"\b[A-Z][A-Z\s]+\s*\(reuters\)\s*-\s*",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"reuters", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"pic\.twitter\.com/\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"http\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"@\w+", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9!?]+", " ", text)
    text = re.sub(r"([!?])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_downsample():
    fake_df = pd.read_csv(FAKE_PATH)
    fake_df["label"] = 1

    true_df = pd.read_csv(TRUE_PATH)
    true_df["label"] = 0

    for df in (fake_df, true_df):
        df["title"] = df["title"].fillna("")
        df["text"] = df["text"].fillna("")
        df["combined_text"] = (df["title"] + " " + df["text"]).str.strip()
        df["clean_text"] = df["combined_text"].apply(clean_text_lstm)

    combined_df = pd.concat([fake_df, true_df], ignore_index=True)
    texts = combined_df["clean_text"].values
    labels = combined_df["label"].values

    if len(texts) < 5000:
        raise ValueError("Dataset must contain at least 5000 samples.")

    X_small, _, y_small, _ = train_test_split(
        texts,
        labels,
        train_size=5000,
        stratify=labels,
        random_state=42,
    )
    return X_small, y_small


def tokenize_and_pad(texts):
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
        sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )
    return padded


def prepare_datasets():
    global X_TRAIN, X_VAL, Y_TRAIN, Y_VAL
    texts, labels = load_and_downsample()
    padded = tokenize_and_pad(texts)
    X_TRAIN, X_VAL, Y_TRAIN, Y_VAL = train_test_split(
        padded,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )


def train_custom_model(
    use_bidirectional: bool = False,
    units: int = 64,
    dropout: float = 0.2,
    embedding_dim: int = 100,
) -> float:
    if any(arr is None for arr in (X_TRAIN, X_VAL, Y_TRAIN, Y_VAL)):
        raise RuntimeError("Datasets not prepared. Call prepare_datasets() first.")

    model = Sequential()
    model.add(
        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=embedding_dim,
            input_length=MAX_LEN,
        )
    )

    lstm_layer = LSTM(units)
    if use_bidirectional:
        model.add(Bidirectional(lstm_layer))
    else:
        model.add(lstm_layer)

    model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_TRAIN,
        Y_TRAIN,
        epochs=3,
        batch_size=64,
        validation_data=(X_VAL, Y_VAL),
        verbose=0,
    )
    return history.history["val_accuracy"][-1]


def run_experiments():
    prepare_datasets()

    experiments = [
        {
            "name": "Baseline",
            "param_change": "None (defaults)",
            "kwargs": dict(
                use_bidirectional=False,
                units=64,
                dropout=0.2,
                embedding_dim=100,
            ),
        },
        {
            "name": "Architecture",
            "param_change": "use_bidirectional=True",
            "kwargs": dict(
                use_bidirectional=True,
                units=64,
                dropout=0.2,
                embedding_dim=100,
            ),
        },
        {
            "name": "Capacity",
            "param_change": "units=32",
            "kwargs": dict(
                use_bidirectional=False,
                units=32,
                dropout=0.2,
                embedding_dim=100,
            ),
        },
        {
            "name": "Regularization",
            "param_change": "dropout=0.5",
            "kwargs": dict(
                use_bidirectional=False,
                units=64,
                dropout=0.5,
                embedding_dim=100,
            ),
        },
        {
            "name": "Representation",
            "param_change": "embedding_dim=50",
            "kwargs": dict(
                use_bidirectional=False,
                units=64,
                dropout=0.2,
                embedding_dim=50,
            ),
        },
    ]

    results = []
    baseline_acc = None

    for exp in experiments:
        acc = train_custom_model(**exp["kwargs"])
        if exp["name"] == "Baseline":
            baseline_acc = acc
        delta = acc - baseline_acc
        results.append(
            {
                "name": exp["name"],
                "param": exp["param_change"],
                "accuracy": acc,
                "delta": delta,
            }
        )

    return results


def display_results(results):
    print(
        f"{'Experiment':<15} | {'Parameter Changed':<30} | {'Accuracy':<10} | {'Delta':<10}"
    )
    print("-" * 75)
    for entry in results:
        print(
            f"{entry['name']:<15} | {entry['param']:<30} | {entry['accuracy']:<10.4f} | {entry['delta']:<10.4f}"
        )

    deltas = [abs(entry["delta"]) for entry in results if entry["name"] != "Baseline"]
    if deltas:
        best_idx = np.argmax(deltas) + 1  # skip baseline
        best_entry = results[best_idx]
        direction = "increase" if best_entry["delta"] > 0 else "decrease"
        print("\nGreatest impact:")
        print(
            f"{best_entry['name']} ({best_entry['param']}) caused a {direction} of {best_entry['delta']:.4f} in validation accuracy."
        )


def main():
    results = run_experiments()
    display_results(results)


if __name__ == "__main__":
    main()

