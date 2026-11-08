from sklearn.model_selection import GridSearchCV, StratifiedKFold

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
