# tests/test_evaluate.py
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluate import compare_models, evaluate, train_vs_cv_scores
from src.models import decision_tree, logistic_regression


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
