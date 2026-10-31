# tests/test_load_data.py
import pytest

from src.load_data import DATA_DIR, load_wine_data, make_binary_target

needs_data = pytest.mark.skipif(
    not (DATA_DIR / "winequality-red.csv").exists(),
    reason="Fichiers CSV absents (définir WINE_DATA_DIR)",
)


@needs_data
def test_shape_and_types():
    df = load_wine_data()
    assert len(df) == 1599 + 4898
    assert set(df["type"]) == {"rouge", "blanc"}


@needs_data
def test_binary_target():
    df = make_binary_target(load_wine_data())
    assert set(df["bon"].unique()) <= {0, 1}