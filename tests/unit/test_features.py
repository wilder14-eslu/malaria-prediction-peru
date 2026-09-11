"""Tests unitarios de feature engineering: grilla semanal, lags, rolling,
tendencia, estacionalidad ciclica y ratios de especie.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.epidemiological import add_species_ratio_features
from src.features.pipeline import build_features
from src.features.temporal import (
    add_cyclical_week_features,
    add_lag_features,
    add_rolling_features,
    add_trend_features,
    complete_weekly_grid,
)


@pytest.fixture
def df_canonical_with_gap() -> pd.DataFrame:
    """Un UBIGEO con semanas 1, 2 y 4 de 2020 (salta la semana 3 a proposito,
    para verificar que complete_weekly_grid la rellena con cases=0).
    """
    return pd.DataFrame(
        {
            "ubigeo": ["160101", "160101", "160101"],
            "departamento": ["LORETO"] * 3,
            "provincia": ["MAYNAS"] * 3,
            "distrito": ["IQUITOS"] * 3,
            "epi_year": [2020, 2020, 2020],
            "epi_week": [1, 2, 4],
            "cases_falciparum": [3, 5, 1],
            "cases_vivax": [7, 15, 1],
            "cases_total": [10, 20, 2],
        }
    )


def test_complete_weekly_grid_fills_missing_week_with_zero(
    df_canonical_with_gap: pd.DataFrame,
) -> None:
    result = complete_weekly_grid(df_canonical_with_gap)

    assert len(result) == 4  # semanas 1, 2, 3 (rellenada), 4
    week_3 = result.loc[result["epi_week"] == 3].iloc[0]
    assert week_3["cases_total"] == 0
    assert week_3["cases_falciparum"] == 0
    assert week_3["cases_vivax"] == 0
    # Los campos estaticos (geografia) se propagan a la semana rellenada.
    assert week_3["distrito"] == "IQUITOS"


def test_lag_1_uses_filled_zero_not_previous_observed_row(
    df_canonical_with_gap: pd.DataFrame,
) -> None:
    """El punto central: lag_1 de la semana 4 debe ser el valor (rellenado)
    de la semana 3, que es 0, NO el valor de la semana 2 (20).
    """
    grid = complete_weekly_grid(df_canonical_with_gap)
    result = add_lag_features(grid, lags=(1,))

    week_4 = result.loc[result["epi_week"] == 4].iloc[0]
    assert week_4["lag_1"] == 0


def test_rolling_mean_excludes_current_week(df_canonical_with_gap: pd.DataFrame) -> None:
    grid = complete_weekly_grid(df_canonical_with_gap)
    result = add_rolling_features(grid, window=4)

    # Semana 4: promedio movil de semanas 1,2,3 (10, 20, 0) = 10, sin incluir
    # el propio valor de la semana 4 (que seria fuga de informacion futura).
    week_4 = result.loc[result["epi_week"] == 4].iloc[0]
    assert week_4["rolling_mean_4"] == pytest.approx(10.0)


def test_growth_rate_is_nan_when_previous_week_has_zero_cases(
    df_canonical_with_gap: pd.DataFrame,
) -> None:
    grid = complete_weekly_grid(df_canonical_with_gap)
    result = add_trend_features(grid)

    # Semana 4 sigue a la semana 3 (rellenada con 0 casos): no hay tasa de
    # crecimiento valida, debe ser NaN y no +inf.
    week_4 = result.loc[result["epi_week"] == 4].iloc[0]
    assert np.isnan(week_4["growth_rate"])


def test_cyclical_week_features_are_bounded(df_canonical_with_gap: pd.DataFrame) -> None:
    result = add_cyclical_week_features(df_canonical_with_gap)
    assert result["week_sin"].between(-1, 1).all()
    assert result["week_cos"].between(-1, 1).all()


def test_species_ratio_is_nan_when_no_cases() -> None:
    df = pd.DataFrame({"cases_falciparum": [0, 3], "cases_vivax": [0, 1], "cases_total": [0, 4]})
    result = add_species_ratio_features(df)

    assert np.isnan(result.loc[0, "vivax_ratio"])
    assert result.loc[1, "falciparum_ratio"] == pytest.approx(0.75)


def test_build_features_pipeline_runs_end_to_end(df_canonical_with_gap: pd.DataFrame) -> None:
    result = build_features(df_canonical_with_gap)

    expected_cols = {
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4",
        "rolling_mean_4",
        "rolling_std_4",
        "growth_rate",
        "week_sin",
        "week_cos",
        "vivax_ratio",
        "falciparum_ratio",
    }
    assert expected_cols.issubset(result.columns)
    assert len(result) == 4
