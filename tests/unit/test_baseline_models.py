"""Tests de los modelos baseline: naive, seasonal naive y media movil."""

from __future__ import annotations

import pandas as pd

from src.models.baseline import predict_moving_average, predict_naive, predict_seasonal_naive


def test_predict_naive_repeats_current_value_for_all_horizons() -> None:
    df = pd.DataFrame({"cases_total": [5, 10, 3]})
    preds = predict_naive(df, horizons=(1, 2))

    assert (preds[1] == df["cases_total"]).all()
    assert (preds[2] == df["cases_total"]).all()


def test_predict_seasonal_naive_uses_same_week_previous_year() -> None:
    # Dos anios de 52 semanas cada uno para el mismo distrito. La semana 1
    # del anio 2 (epi_index = 2*52+1) deberia pronosticar (h=1) con el valor
    # de epi_index = 2*52+1-52+1 = semana 2 del anio 1.
    weeks = list(range(1, 53))
    df = pd.DataFrame(
        {
            "ubigeo": ["A"] * 104,
            "epi_index": [1 * 52 + w for w in weeks] + [2 * 52 + w for w in weeks],
            "cases_total": list(range(52)) + list(range(100, 152)),
        }
    )

    preds = predict_seasonal_naive(df, horizons=(1,))

    # Fila con epi_index = 2*52+1 (primera semana del segundo anio, valor 100).
    target_row = df.loc[df["epi_index"] == 2 * 52 + 1].index[0]
    # Debe apuntar al valor de epi_index = 2*52+1-52+1 = 1*52+2 -> cases_total=1
    assert preds[1].loc[target_row] == 1


def test_predict_moving_average_uses_precomputed_rolling_column() -> None:
    df = pd.DataFrame({"rolling_mean_4": [1.5, 2.5, 3.5]})
    preds = predict_moving_average(df, horizons=(1, 3), window=4)

    assert (preds[1] == df["rolling_mean_4"]).all()
    assert (preds[3] == df["rolling_mean_4"]).all()
