from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data_to_df(path: Union[Path, str]) -> pd.DataFrame:
    """Read a .dat dataset into a two-column pandas DataFrame.

    Inputs:
        path: Filesystem path to a whitespace-separated dialog-act dataset.
    Returns:
        DataFrame with columns `label` and `text` containing cleaned rows.
    """
    data_path = Path(path)
    rows: List[Tuple[str, str]] = []

    with data_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            pieces = line.split(maxsplit=1)
            if len(pieces) != 2:
                # Skip malformed rows to avoid downstream errors.
                continue
            label, text = pieces
            rows.append((label, text))

    return pd.DataFrame(rows, columns=["label", "text"])


def stratified_split(
    df: pd.DataFrame, test_size: float = 0.15, seed: Union[int, None] = 42
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Create stratified train/test splits for text and label columns.

    Inputs:
        df: DataFrame containing at least `text` and `label` columns.
        test_size: Fraction of samples to allocate to the test split.
        seed: Deterministic random state to reproduce splits.
    Returns:
        Tuple containing (x_train, x_test, y_train, y_test) lists.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )
    return x_train, x_test, y_train, y_test


def vectorize_fit_transform(
    vectorizer: CountVectorizer,
    x_train: Sequence[str],
    x_test: Sequence[str],
):
    """Fit the vectorizer on training texts and transform both splits.

    Inputs:
        vectorizer: CountVectorizer (or compatible) instance to train.
        x_train: Iterable of training utterances.
        x_test: Iterable of test utterances.
    Returns:
        Tuple of sparse matrices (x_train_transformed, x_test_transformed).
    """
    x_train_transformed = vectorizer.fit_transform(x_train)
    x_test_transformed = vectorizer.transform(x_test)
    return x_train_transformed, x_test_transformed


def encode_labels(
    encoder: LabelEncoder,
    y_train: Sequence[str],
    y_test: Sequence[str],
):
    """Fit the label encoder on training labels and transform both splits.

    Inputs:
        encoder: LabelEncoder (or compatible) instance to train.
        y_train: Iterable of training label strings.
        y_test: Iterable of test label strings.
    Returns:
        Tuple of numpy arrays (y_train_encoded, y_test_encoded).
    """
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    return y_train_encoded, y_test_encoded


def summarize_labels(
    y_train: Iterable[str],
    y_test: Iterable[str],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Print and return class distribution statistics for each split.

    Inputs:
        y_train: Iterable of training labels.
        y_test: Iterable of test labels.
    Returns:
        Tuple of dictionaries summarizing counts for train and test labels.
    """
    print("--## Dataset Summary ##--")
    y_train_list = list(y_train)
    y_test_list = list(y_test)
    train_counts = dict(
        sorted(Counter(y_train_list).items(), key=lambda kv: kv[1], reverse=True)
    )
    test_counts = dict(
        sorted(Counter(y_test_list).items(), key=lambda kv: kv[1], reverse=True)
    )

    print(
        f"Train: total={len(y_train_list)}, unique={len(train_counts)}, counts={train_counts}"
    )
    print(
        f"Test:  total={len(y_test_list)}, unique={len(test_counts)}, counts={test_counts}"
    )
    print("-----#####-----\n")

    return train_counts, test_counts


def prepare_dataset(path: Union[Path, str]):
    """Preprocess the dataset into vectorized splits ready for training.

    Inputs:
        path: Filesystem path to the raw dialog-act dataset (.dat format).
    Returns:
        Dictionary containing train/test features, encoded labels, and fitted encoders.
    """
    df = load_data_to_df(path)

    label_counts = df["label"].value_counts()
    rare_labels = label_counts[label_counts < 2].index.tolist()

    rare_df = df[df["label"].isin(rare_labels)]
    major_df = df[~df["label"].isin(rare_labels)]

    if not major_df.empty:
        x_train_major, x_test, y_train_major, y_test = stratified_split(major_df, test_size=0.15)
    else:
        x_train_major, x_test, y_train_major, y_test = [], [], [], []

    x_train = list(x_train_major) + rare_df["text"].tolist()
    y_train = list(y_train_major) + rare_df["label"].tolist()

    if rare_labels:
        print(
            "Moved {count} sample(s) from rare label(s) {labels} into the train split only.".format(
                count=len(rare_df), labels=rare_labels
            )
        )

    vectorizer = CountVectorizer()
    x_train_transformed, x_test_transformed = vectorize_fit_transform(
        vectorizer, x_train, x_test
    )

    encoder = LabelEncoder()
    y_train_encoded, y_test_encoded = encode_labels(encoder, y_train, y_test)

    summarize_labels(y_train, y_test)

    return {
        "x_train": x_train_transformed,
        "x_test": x_test_transformed,
        "y_train": y_train_encoded,
        "y_test": y_test_encoded,
        "encoder": encoder,
        "vectorizer": vectorizer,
    }


if __name__ == "__main__":
    prepare_dataset("datasets/dialog_acts_deduplicated.dat")
