import numpy as np
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression


def correlation_filter(X, threshold=0.8):
    """Renvoie les colonnes à retirer parce qu'elles sont corrélées (|r| > seuil) à une autre."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    return [col for col in upper.columns if (upper[col] > threshold).any()]


def select_kbest(X, y, k=6):
    """Filtre univarié : ANOVA F-value (f_classif)."""
    selector = SelectKBest(f_classif, k=k).fit(X, y)
    return list(X.columns[selector.get_support()])


def select_rfe(X, y, n_features=6):
    """Wrapper : élimination récursive avec régression logistique (données déjà standardisées)."""
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    rfe = RFE(model, n_features_to_select=n_features).fit(X, y)
    return list(X.columns[rfe.support_])