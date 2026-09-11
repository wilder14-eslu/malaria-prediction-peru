"""Tests de las metricas de evaluacion de forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.forecasting import interval_coverage, mae, pinball_loss, rmse, wape


def test_mae_and_rmse_basic() -> None:
    y_true = pd.Series([10, 20, 30])
    y_pred = pd.Series([12, 18, 33])

    assert mae(y_true, y_pred) == pytest.approx((2 + 2 + 3) / 3)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((4 + 4 + 9) / 3))


def test_metrics_ignore_rows_with_nan() -> None:
    y_true = pd.Series([10, np.nan, 30])
    y_pred = pd.Series([12, 18, 33])

    # Solo deben quedar 2 pares validos: (10,12) y (30,33).
    assert mae(y_true, y_pred) == pytest.approx((2 + 3) / 2)


def test_wape_is_nan_when_all_true_values_are_zero() -> None:
    y_true = pd.Series([0, 0, 0])
    y_pred = pd.Series([1, 2, 3])

    assert np.isnan(wape(y_true, y_pred))


def test_wape_normalizes_by_total_true_magnitude() -> None:
    y_true = pd.Series([10, 10])
    y_pred = pd.Series([12, 8])

    assert wape(y_true, y_pred) == pytest.approx((2 + 2) / 20)


def test_pinball_loss_penalizes_asymmetrically() -> None:
    # Para el cuantil 0.9, sub-predecir (quedarse corto) pesa mas que
    # sobre-predecir en la misma magnitud.
    y_true = pd.Series([100])

    under_prediction = pinball_loss(y_true, pd.Series([80]), quantile=0.9)
    over_prediction = pinball_loss(y_true, pd.Series([120]), quantile=0.9)

    assert under_prediction > over_prediction


def test_interval_coverage_all_within_bounds() -> None:
    y_true = pd.Series([5, 10, 15])
    lower = pd.Series([0, 5, 10])
    upper = pd.Series([10, 15, 20])

    assert interval_coverage(y_true, lower, upper) == pytest.approx(1.0)


def test_interval_coverage_partial() -> None:
    y_true = pd.Series([5, 100, 15])
    lower = pd.Series([0, 5, 10])
    upper = pd.Series([10, 15, 20])

    assert interval_coverage(y_true, lower, upper) == pytest.approx(2 / 3)
