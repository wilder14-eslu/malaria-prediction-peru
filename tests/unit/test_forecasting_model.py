"""Tests del modelo de forecasting por cuantiles (LightGBM)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.forecasting import get_feature_columns, predict_quantiles, train_quantile_models


def test_get_feature_columns_excludes_identifiers_and_targets() -> None:
    df = pd.DataFrame(
        columns=[
            "ubigeo",
            "departamento",
            "provincia",
            "distrito",
            "epi_year",
            "epi_week",
            "epi_index",
            "lag_1",
            "target_h1",
            "target_h2",
        ]
    )
    cols = get_feature_columns(df, horizons=(1, 2))
    assert cols == ["lag_1"]


def test_predict_quantiles_returns_non_decreasing_quantiles() -> None:
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    y = 3 * x + rng.normal(scale=2, size=n)
    df = pd.DataFrame({"feat": x, "target": y})
    feature_cols = ["feat"]

    models = train_quantile_models(
        df,
        feature_cols,
        target_col="target",
        quantiles=(0.1, 0.5, 0.9),
        params={"n_estimators": 30, "num_leaves": 7, "min_child_samples": 5},
    )
    preds = predict_quantiles(models, df, feature_cols)

    assert (preds["q0.1"] <= preds["q0.5"]).all()
    assert (preds["q0.5"] <= preds["q0.9"]).all()


def test_train_quantile_models_drops_rows_with_nan_target() -> None:
    df = pd.DataFrame({"feat": [1.0, 2.0, 3.0, 4.0], "target": [10.0, 20.0, np.nan, 40.0]})

    # No debe fallar aunque haya un NaN en el target: se descarta esa fila.
    models = train_quantile_models(
        df,
        feature_cols=["feat"],
        target_col="target",
        quantiles=(0.5,),
        params={"n_estimators": 5, "num_leaves": 3, "min_child_samples": 1},
    )
    assert 0.5 in models
