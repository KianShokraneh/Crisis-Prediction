from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

SENTIMENT140_ZIP = "sentiment140.zip"
TRAIN_FILENAME = "training.1600000.processed.noemoticon.csv"


@dataclass
class Sentiment140Split:
    train_texts: list[str]
    train_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]


def ensure_training_csv(base_path: str, zip_path: str | None = None) -> str:
    train_csv = os.path.join(base_path, TRAIN_FILENAME)
    if os.path.exists(train_csv):
        return train_csv

    if zip_path is None:
        zip_path = os.path.join(base_path, SENTIMENT140_ZIP)

    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Missing {TRAIN_FILENAME}. Expected at {train_csv} or inside {zip_path}."
        )

    os.makedirs(base_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        if TRAIN_FILENAME not in zf.namelist():
            raise FileNotFoundError(f"{TRAIN_FILENAME} not found in {zip_path}.")
        zf.extract(TRAIN_FILENAME, base_path)
    return train_csv


def load_sentiment140(train_csv: str, limit: int | None = None) -> pd.DataFrame:
    # Columns: target, ids, date, flag, user, text
    df = pd.read_csv(
        train_csv,
        header=None,
        encoding="latin-1",
        usecols=[0, 5],
        names=["label", "text"],
    )
    # Map labels: 0 = negative, 4 = positive
    df = df[df["label"].isin([0, 4])].copy()
    df["label"] = df["label"].map({0: 0, 4: 1})
    if limit:
        df = df.head(limit)
    df["text"] = df["text"].astype(str).str.replace("\n", " ").str.replace("\t", " ")
    return df


def train_test_split_df(
    df: pd.DataFrame,
    train_size: int | None,
    test_size: int | None,
    seed: int,
) -> Sentiment140Split:
    if train_size is None and test_size is None:
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=seed, stratify=df["label"]
        )
    else:
        if test_size is None:
            test_size = max(1, int(0.2 * len(df)))
        if train_size is None:
            train_size = len(df) - test_size
        subset_size = min(len(df), train_size + test_size)
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=subset_size, random_state=seed)
        subset_idx, _ = next(splitter.split(df, df["label"]))
        df_subset = df.iloc[subset_idx]
        train_df, test_df = train_test_split(
            df_subset,
            train_size=train_size,
            test_size=test_size,
            random_state=seed,
            stratify=df_subset["label"],
        )

    return Sentiment140Split(
        train_texts=train_df["text"].tolist(),
        train_labels=train_df["label"].tolist(),
        test_texts=test_df["text"].tolist(),
        test_labels=test_df["label"].tolist(),
    )
