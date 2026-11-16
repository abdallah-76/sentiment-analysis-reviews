"""Chargement et préparation des données.

Source par défaut : avis Amazon (anglais) hébergés sur Hugging Face.
Les notes de 1 à 5 étoiles sont regroupées en 3 classes :
    1-2 étoiles -> négatif (0) | 3 étoiles -> neutre (1) | 4-5 étoiles -> positif (2)

Si Hugging Face n'est pas accessible, on peut passer son propre CSV
(colonnes `text` et `label`, avec label = 0/1/2 ou négatif/neutre/positif).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["négatif", "neutre", "positif"]
DATASET_ID = "SetFit/amazon_reviews_multi_en"
ROOT = Path(__file__).resolve().parent.parent


def stars_to_sentiment(label: int) -> int:
    """Dans le dataset, label 0..4 correspond à 1..5 étoiles."""
    if label <= 1:
        return 0
    if label == 2:
        return 1
    return 2


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    return df[df["text"] != ""].reset_index(drop=True)


def _from_hf(train_size: int | None, seed: int):
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID)
    frames = {}
    for split in ("train", "validation", "test"):
        df = ds[split].to_pandas()[["text", "label"]]
        df["label"] = df["label"].map(stars_to_sentiment)
        frames[split] = _clean(df)

    train = frames["train"]
    if train_size and train_size < len(train):
        train = train.sample(n=train_size, random_state=seed).reset_index(drop=True)
    return train, frames["validation"], frames["test"]


def _from_csv(path: str, train_size: int | None, seed: int):
    df = pd.read_csv(path)
    if not {"text", "label"} <= set(df.columns):
        raise ValueError("Le CSV doit contenir les colonnes 'text' et 'label'.")
    if df["label"].dtype == object:
        df["label"] = df["label"].map({name: i for i, name in enumerate(LABELS)})
    df = _clean(df)
    df["label"] = df["label"].astype(int)

    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=seed)
    if train_size and train_size < len(train):
        train = train.sample(n=train_size, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_splits(csv: str | None = None, train_size: int | None = None, seed: int = 42):
    """Retourne (train, validation, test) : DataFrames avec colonnes text, label."""
    if csv:
        return _from_csv(csv, train_size, seed)
    return _from_hf(train_size, seed)
