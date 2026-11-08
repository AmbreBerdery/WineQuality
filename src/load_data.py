import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Dossier data/ à la racine du projet (WineQuality/raw/data/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR = Path(os.environ.get("WINE_DATA_DIR", DEFAULT_DATA_DIR))

RED_FILE = "winequality-red.csv" #je rajoute ca sinon ca fait un pb avec le path et il comprend pas ou trouver la donnée
WHITE_FILE = "winequality-white.csv"

TARGET_THRESHOLD = 7   # bon vin si la qualite >= 7
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_wine_data(data_dir=DATA_DIR):
    """Charge les vins rouges et blancs et les fusionne avec une colonne 'type'."""
    data_dir = Path(data_dir)

    for name in (RED_FILE, WHITE_FILE):
        if not (data_dir / name).exists():
            raise FileNotFoundError(
                f"{name} introuvable dans {data_dir.resolve()}. "
                "Placez les fichiers dans WineQuality/data/ ou définissez WINE_DATA_DIR."
            )

    red = pd.read_csv(data_dir / RED_FILE, sep=";")
    white = pd.read_csv(data_dir / WHITE_FILE, sep=";")
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