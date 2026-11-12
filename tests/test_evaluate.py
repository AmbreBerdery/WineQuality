# tests/test_models.py
# Tests des modèles k-NN, SVM et Naive Bayes, et des recherches d'hyperparamètres.
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.models import KNN_GRID, SVM_DISTRIBUTIONS, knn, naive_bayes, svm
from src.tuning import run_grid_search, run_random_search


@pytest.fixture
def data():
    """Petit jeu déséquilibré (environ 20 % de positifs), comme les vins."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"] + 0.5 * rng.normal(size=300) > 0.8).astype(int))
    return X, y


CV3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


# --- Construction des pipelines ---

@pytest.mark.parametrize("factory", [knn, svm, naive_bayes])
def test_pipeline_contient_pretraitement_et_modele(factory):
    pipe = factory()
    assert list(pipe.named_steps) == ["prep", "model"]


@pytest.mark.parametrize("factory", [knn, svm, naive_bayes])
def test_selection_de_variables(factory, data):
    """Le modèle ne doit voir que les variables sélectionnées."""
    X, y = data
    pipe = factory(["a", "b"]).fit(X, y)
    assert pipe.named_steps["model"].n_features_in_ == 2


@pytest.mark.parametrize("factory", [knn, svm, naive_bayes])
def test_predictions_binaires(factory, data):
    X, y = data
    pred = factory().fit(X, y).predict(X)
    assert len(pred) == len(y)
    assert set(np.unique(pred)) <= {0, 1}


# --- Réglages propres à chaque modèle ---

def test_svm_compense_le_desequilibre():
    assert svm().named_steps["model"].class_weight == "balanced"


def test_svm_fournit_un_score_pour_la_roc(data):
    """Sans probabilités, le SVM doit fournir decision_function (utilisée pour la ROC)."""
    X, y = data
    scores = svm().fit(X, y).decision_function(X)
    assert scores.shape == (len(y),)


def test_naive_bayes_priors_egaux(data):
    X, y = data
    pipe = naive_bayes(priors=[0.5, 0.5]).fit(X, y)
    assert np.allclose(pipe.named_steps["model"].class_prior_, [0.5, 0.5])


def test_naive_bayes_priors_egaux_augmentent_le_rappel(data):
    """Des priors égaux font prédire « positif » au moins aussi souvent que les fréquences."""
    X, y = data
    standard = naive_bayes().fit(X, y).predict(X).sum()
    equilibre = naive_bayes(priors=[0.5, 0.5]).fit(X, y).predict(X).sum()
    assert equilibre >= standard


def test_knn_un_voisin_memorise_le_train(data):
    """Avec k = 1, chaque point du train est son propre voisin : score parfait sur le train."""
    X, y = data
    pipe = knn().set_params(model__n_neighbors=1).fit(X, y)
    assert (pipe.predict(X) == y).all()


def test_grille_knn_valeurs_de_k_impaires():
    """k impair pour éviter les égalités de vote entre deux classes."""
    assert all(k % 2 == 1 for k in KNN_GRID["model__n_neighbors"])


def test_espace_svm_gamma_seulement_pour_rbf():
    """gamma n'a pas de sens pour le noyau linéaire."""
    for espace in SVM_DISTRIBUTIONS:
        if espace["model__kernel"] == ["linear"]:
            assert "model__gamma" not in espace


# --- Recherches d'hyperparamètres ---

def test_grid_search_knn(data):
    X, y = data
    grid = {"model__n_neighbors": [3, 5], "model__weights": ["uniform"], "model__p": [2]}
    search = run_grid_search(knn(), grid, X, y, cv=CV3)
    assert search.best_params_["model__n_neighbors"] in [3, 5]
    assert len(search.cv_results_["params"]) == 2
    assert 0 <= search.best_score_ <= 1


def test_random_search_svm_respecte_le_budget(data):
    X, y = data
    search = run_random_search(svm(), SVM_DISTRIBUTIONS, X, y, n_iter=4, cv=CV3)
    assert len(search.cv_results_["params"]) == 4
    assert search.best_params_["model__kernel"] in ["rbf", "linear"]


def test_random_search_reproductible(data):
    """Le random_state commun rend les tirages identiques d'une exécution à l'autre."""
    X, y = data
    a = run_random_search(svm(), SVM_DISTRIBUTIONS, X, y, n_iter=3, cv=CV3)
    b = run_random_search(svm(), SVM_DISTRIBUTIONS, X, y, n_iter=3, cv=CV3)
    assert a.cv_results_["params"] == b.cv_results_["params"]