import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def clean_data(df):
    """Supprime les doublons et les valeurs manquantes, encode 'type' en 0/1."""
    n_before = len(df)
    df = df.drop_duplicates().dropna().copy()
    print(f"Lignes supprimées (doublons ou NA) : {n_before - len(df)}")
    df["is_rouge"] = (df["type"] == "rouge").astype(int)
    return df.drop(columns="type")


class IQRClipper(BaseEstimator, TransformerMixin):
    """Plafonne les valeurs aberrantes aux bornes IQR calculées sur le train.

    On plafonne au lieu de supprimer : on ne perd pas de lignes.
    """

    def __init__(self, k=1.5):
        self.k = k

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        q1, q3 = np.percentile(values, [25, 75], axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        return self

    def transform(self, X):
        values = np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(values, index=X.index, columns=X.columns)
        return values


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Garde uniquement les colonnes retenues par la sélection de variables."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X if self.columns is None else X[self.columns]


def build_preprocessor(selected_features=None):
    """Pipeline réutilisable : plafonnement IQR, standardisation, sélection.

    B et C utilisent ce pipeline tel quel. Il est entraîné uniquement sur le train.
    """
    return Pipeline([
        ("clip", IQRClipper()),
        ("scale", StandardScaler().set_output(transform="pandas")),
        ("select", ColumnSelector(selected_features)),
    ])