import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_validate, learning_curve, validation_curve

from src.tuning import CV, SCORING

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def _scores(model, X):
    """Score continu pour la ROC : probabilité de la classe 1, sinon decision_function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model, X, y):
    """Évalue un modèle déjà entraîné. Fonction commune à tout le groupe.

    Renvoie un dict : accuracy, precision, recall, f1, roc_auc,
    la matrice de confusion et les points (fpr, tpr) de la courbe ROC.
    """
    y_pred = model.predict(X)
    y_score = _scores(model, X)
    fpr, tpr, _ = roc_curve(y, y_score)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_score),
        "confusion_matrix": confusion_matrix(y, y_pred),
        "fpr": fpr,
        "tpr": tpr,
    }


def compare_models(results):
    """Tableau comparatif à partir de {nom: résultat de evaluate()}, trié par F1."""
    table = pd.DataFrame({name: {m: r[m] for m in METRICS} for name, r in results.items()}).T
    return table.sort_values("f1", ascending=False)


def plot_confusion_matrix(result, title="", ax=None):
    ax = ax or plt.gca()
    ConfusionMatrixDisplay(result["confusion_matrix"], display_labels=["moins bon", "bon"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set(title=title, xlabel="Classe prédite", ylabel="Classe réelle")
    ax.grid(False)
    return ax


def plot_roc_curves(results, ax=None):
    """Superpose les courbes ROC de plusieurs modèles ({nom: résultat de evaluate()})."""
    ax = ax or plt.gca()
    for name, r in results.items():
        ax.plot(r["fpr"], r["tpr"], label=f"{name} (AUC = {r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Hasard (AUC = 0.5)")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs (rappel)")
    ax.set_title("Courbes ROC")
    ax.legend(loc="lower right")
    return ax


def train_vs_cv_scores(model, X, y, scoring=SCORING, cv=CV):
    """Biais/variance : score moyen sur les plis d'entraînement vs de validation.

    Score train bas => biais élevé (sous-apprentissage).
    Écart train - validation grand => variance élevée (surapprentissage).
    """
    res = cross_validate(model, X, y, scoring=scoring, cv=cv, return_train_score=True)
    train, val = res["train_score"], res["test_score"]
    return {
        "train": train.mean(),
        "validation": val.mean(),
        "validation_std": val.std(),
        "ecart": train.mean() - val.mean(),
    }


def _plot_train_val(ax, x, train, val, xlabel, title, scoring):
    for scores, label in [(train, "Entraînement"), (val, "Validation croisée")]:
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(x, mean, "o-", label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(scoring)
    ax.set_title(title)
    ax.legend()
    return ax


def plot_learning_curve(model, X, y, title="", scoring=SCORING, cv=CV, ax=None):
    """Score train et validation en fonction de la taille du jeu d'entraînement."""
    ax = ax or plt.gca()
    sizes, train, val = learning_curve(
        model, X, y, cv=cv, scoring=scoring, train_sizes=np.linspace(0.1, 1.0, 8)
    )
    return _plot_train_val(ax, sizes, train, val, "Taille du jeu d'entraînement", title, scoring)


def plot_validation_curve(model, X, y, param_name, param_range, title="", scoring=SCORING, cv=CV, ax=None):
    """Score train et validation en fonction d'un hyperparamètre (complexité du modèle)."""
    ax = ax or plt.gca()
    train, val = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring=scoring
    )
    return _plot_train_val(ax, param_range, train, val, param_name, title, scoring)
