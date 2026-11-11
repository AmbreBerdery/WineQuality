import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.load_data import PROJECT_ROOT, RANDOM_STATE
from src.preprocess import build_preprocessor

from scipy.stats import loguniform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

MODELS_DIR = PROJECT_ROOT / "models"


def load_selected_features():
    """Variables retenues par la sélection (fichier produit par le notebook 02)."""
    with open(MODELS_DIR / "selected_features.json", encoding="utf-8") as f:
        return json.load(f)


def make_pipeline(model, selected_features=None):
    """Prétraitement commun + modèle. Les hyperparamètres du modèle se nomment model__<param>."""
    return Pipeline([
        ("prep", build_preprocessor(selected_features)),
        ("model", model),
    ])


def save_model(pipeline, name):
    """Sauvegarde dans models/<name>.joblib : le notebook 04 charge tous ces fichiers."""
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / f"{name}.joblib")


# --- Régression logistique (modèle linéaire) ---

def logistic_regression(selected_features=None):
    model = LogisticRegression(class_weight="balanced", solver="saga", max_iter=5000, random_state=RANDOM_STATE)
    return make_pipeline(model, selected_features)


# Pénalité via l1_ratio (scikit-learn >= 1.8) : 0 = L2 (ridge), 1 = L1 (lasso), 0.5 = elastic-net
LOGREG_GRID = {
    "model__C": np.logspace(-3, 2, 11),
    "model__l1_ratio": [0.0, 0.5, 1.0],
}


# --- Arbre de décision (modèle non linéaire) ---

def decision_tree(selected_features=None):
    model = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    return make_pipeline(model, selected_features)


TREE_GRID = {
    "model__max_depth": [2, 3, 4, 5, 6, 8, 10, 12, 15, None],
    "model__min_samples_leaf": [1, 5, 10, 20, 50, 100],
}




# --- k-NN (modèle à distance) ---

def knn(selected_features=None):
    return make_pipeline(KNeighborsClassifier(), selected_features)


KNN_GRID = {
    "model__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51],
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2],   # 1 = Manhattan, 2 = euclidienne
}


# --- SVM (modèle à noyau) ---

def svm(selected_features=None):
    model = SVC(class_weight="balanced", random_state=RANDOM_STATE)
    return make_pipeline(model, selected_features)


SVM_DISTRIBUTIONS = [
    {"model__kernel": ["rbf"],
     "model__C": loguniform(1e-2, 1e3),
     "model__gamma": loguniform(1e-4, 1e1)},
    {"model__kernel": ["linear"],
     "model__C": loguniform(1e-2, 1e2)},
]