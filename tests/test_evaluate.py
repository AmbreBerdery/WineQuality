# tests/test_evaluate.py
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluate import compare_models, evaluate, train_vs_cv_scores
from src.models import decision_tree, knn, logistic_regression, naive_bayes, svm


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"] + 0.5 * rng.normal(size=300) > 0.8).astype(int))
    return X, y


def test_evaluate_keys_and_ranges(data):
    X, y = data
    result = evaluate(LogisticRegression().fit(X, y), X, y)
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert 0 <= result[metric] <= 1
    assert result["confusion_matrix"].sum() == len(y)


def test_evaluate_perfect_model():
    X = pd.DataFrame({"a": [0.0, 0.1, 0.2, 1.0, 1.1, 1.2]})
    y = pd.Series([0, 0, 0, 1, 1, 1])
    result = evaluate(LogisticRegression(C=100).fit(X, y), X, y)
    assert result["f1"] == 1 and result["roc_auc"] == 1


def test_compare_models_sorted_by_f1(data):
    X, y = data
    results = {
        "logreg": evaluate(logistic_regression().fit(X, y), X, y),
        "arbre": evaluate(decision_tree().fit(X, y), X, y),
    }
    table = compare_models(results)
    assert list(table.columns) == ["accuracy", "precision", "recall", "f1", "roc_auc"]
    assert table["f1"].is_monotonic_decreasing


def test_unpruned_tree_overfits(data):
    X, y = data
    scores = train_vs_cv_scores(decision_tree(), X, y)
    assert scores["train"] == pytest.approx(1.0)
    assert scores["ecart"] > 0


# --- k-NN, Naive Bayes et SVM ---

@pytest.mark.parametrize("factory", [knn, naive_bayes, svm])
def test_evaluate_knn_nb_svm(factory, data):
    """evaluate() fonctionne pour les trois modèles et renvoie des métriques valides."""
    X, y = data
    result = evaluate(factory().fit(X, y), X, y)
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert 0 <= result[metric] <= 1
    assert result["confusion_matrix"].sum() == len(y)


def test_evaluate_svm_sans_probabilites(data):
    """Le SVM n'utilise pas probability=True : la ROC repose sur decision_function."""
    X, y = data
    model = svm().fit(X, y)
    assert model.named_steps["model"].probability is False
    result = evaluate(model, X, y)
    assert result["roc_auc"] > 0.5


def test_compare_models_cinq_modeles(data):
    """Le tableau comparatif contient les cinq modèles, triés par F1."""
    X, y = data
    factories = {
        "naive_bayes": naive_bayes,
        "regression_logistique": logistic_regression,
        "arbre_decision": decision_tree,
        "knn": knn,
        "svm": svm,
    }
    results = {name: evaluate(f().fit(X, y), X, y) for name, f in factories.items()}
    table = compare_models(results)
    assert set(table.index) == set(factories)
    assert table["f1"].is_monotonic_decreasing


def test_knn_un_voisin_overfits(data):
    """Avec k = 1, chaque vin du train est son propre voisin : F1 de 1 en entraînement."""
    X, y = data
    scores = train_vs_cv_scores(knn().set_params(model__n_neighbors=1), X, y)
    assert scores["train"] == pytest.approx(1.0)
    assert scores["ecart"] > 0


def test_naive_bayes_ne_surapprend_pas(data):
    """Naive Bayes a peu de paramètres : écart train / validation faible."""
    X, y = data
    scores = train_vs_cv_scores(naive_bayes(), X, y)
    assert abs(scores["ecart"]) < 0.1


def test_naive_bayes_priors_egaux_augmentent_le_rappel(data):
    """Des priors égaux compensent le déséquilibre : le rappel ne baisse pas."""
    X, y = data
    standard = evaluate(naive_bayes().fit(X, y), X, y)
    equilibre = evaluate(naive_bayes(priors=[0.5, 0.5]).fit(X, y), X, y)
    assert equilibre["recall"] >= standard["recall"]