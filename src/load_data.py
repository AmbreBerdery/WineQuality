import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_DIR = Path(os.environ.get("WINE_DATA_DIR", DEFAULT_DATA_DIR))

TARGET_THRESHOLD = 7   # bon vin si qualite >= 7
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_wine_data(data_dir=DATA_DIR):
    """Charge les vins rouges et blancs et les fusionne avec une colonne 'type'."""
    data_dir = Path(data_dir)
    red = pd.read_csv(data_dir / "winequality-red.csv", sep=";")
    white = pd.read_csv(data_dir / "winequality-white.csv", sep=";")
    red["type"] = "rouge"
    white["type"] = "blanc"
    return pd.concat([red, white], ignore_index=True)


def make_binary_target(df):
    """Ajoute la cible binaire 'bon' (1 si qualite >= 7). Seule définition du projet."""
    df = df.copy()
    df["bon"] = (df["quality"] >= TARGET_THRESHOLD).astype(int)
    return df


def split_data(df):
    """Découpage train/test stratifié, identique pour tout le groupe."""
    X = df.drop(columns=["quality", "bon"])
    y = df["bon"]
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )