from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from src.load_data import RANDOM_STATE

# Cadre de validation commun à tout le groupe : mêmes plis, même métrique.
# F1 plutôt que l'accuracy car les bons vins ne représentent qu'environ 19 % des données.
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = "f1"


def run_grid_search(pipeline, param_grid, X, y, scoring=SCORING, cv=CV):
    """Grid Search avec validation croisée k-fold, sur le train uniquement.

    Les scores train sont conservés pour l'analyse biais/variance.
    """
    search = GridSearchCV(
        pipeline, param_grid, scoring=scoring, cv=cv, n_jobs=-1, return_train_score=True
    )
    return search.fit(X, y)



def run_random_search(pipeline, param_distributions, X, y, n_iter=60, scoring=SCORING, cv=CV):
    """Random Search : n_iter combinaisons tirées au hasard dans les distributions.

    Même CV et même métrique que run_grid_search, pour des résultats comparables.
    """
    search = RandomizedSearchCV(
        pipeline, param_distributions, n_iter=n_iter, scoring=scoring, cv=cv,
        n_jobs=-1, return_train_score=True, random_state=RANDOM_STATE,
    )
    return search.fit(X, y)
