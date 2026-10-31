# tests/test_preprocess.py
import numpy as np
import pandas as pd

from src.feature_selection import correlation_filter
from src.preprocess import IQRClipper, build_preprocessor


def test_iqr_clipper_caps_outlier():
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5, 1000]})
    clipped = IQRClipper().fit_transform(X)
    assert clipped["a"].max() < 1000


def test_preprocessor_is_standardized():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(5, 3, size=(200, 3)), columns=["a", "b", "c"])
    out = build_preprocessor().fit_transform(X)
    assert np.allclose(out.mean(), 0, atol=1e-7)
    assert np.allclose(out.std(ddof=0), 1, atol=1e-7)


def test_correlation_filter_drops_duplicate():
    rng = np.random.default_rng(0)
    a = rng.normal(size=100)
    X = pd.DataFrame({"a": a, "a_copy": a * 2 + 1e-6, "b": rng.normal(size=100)})
    assert correlation_filter(X, threshold=0.9) == ["a_copy"]