"""Tests de agregacion y union de features climaticas (Etapa 2)."""

from __future__ import annotations

import pandas as pd

from src.features.climate import add_climate_features, aggregate_climate_to_weekly

DAILY_CLIMATE_CSV = """departamento,date,t2m,t2m_max,t2m_min,precip_mm,rh2m
LORETO,2024-01-01,26.0,30.0,22.0,5.0,80.0
LORETO,2024-01-02,27.0,31.0,23.0,10.0,82.0
LORETO,2024-01-08,25.0,29.0,21.0,2.0,78.0
TUMBES,2024-01-01,28.0,33.0,24.0,0.0,70.0
"""


def test_aggregate_climate_to_weekly_groups_by_iso_week() -> None:
    df_daily = pd.read_csv(pd.io.common.StringIO(DAILY_CLIMATE_CSV))

    result = aggregate_climate_to_weekly(df_daily)

    # 2024-01-01 y 2024-01-02 son lunes/martes de la semana ISO 1 de 2024;
    # 2024-01-08 es lunes de la semana ISO 2.
    loreto_w1 = result.loc[
        (result["departamento"] == "LORETO")
        & (result["epi_year"] == 2024)
        & (result["epi_week"] == 1)
    ].iloc[0]
    assert loreto_w1["t2m_mean"] == 26.5
    assert loreto_w1["precip_mm_sum"] == 15.0
    assert loreto_w1["t2m_max"] == 31.0
    assert loreto_w1["t2m_min"] == 22.0

    loreto_w2 = result.loc[
        (result["departamento"] == "LORETO")
        & (result["epi_year"] == 2024)
        & (result["epi_week"] == 2)
    ].iloc[0]
    assert loreto_w2["precip_mm_sum"] == 2.0

    assert len(result) == 3  # LORETO w1, LORETO w2, TUMBES w1


def test_add_climate_features_left_joins_by_department_and_week() -> None:
    df_daily = pd.read_csv(pd.io.common.StringIO(DAILY_CLIMATE_CSV))
    df_climate_weekly = aggregate_climate_to_weekly(df_daily)

    df_features = pd.DataFrame(
        {
            "ubigeo": ["160101", "240101"],
            "departamento": ["LORETO", "TUMBES"],
            "epi_year": [2024, 2024],
            "epi_week": [1, 1],
            "cases_total": [10, 3],
        }
    )

    result = add_climate_features(df_features, df_climate_weekly)

    assert len(result) == 2  # no duplica filas de casos
    loreto_row = result.loc[result["ubigeo"] == "160101"].iloc[0]
    assert loreto_row["t2m_mean"] == 26.5


def test_add_climate_features_keeps_rows_with_no_matching_climate() -> None:
    df_climate_weekly = pd.DataFrame(
        {
            "departamento": ["LORETO"],
            "epi_year": [2024],
            "epi_week": [1],
            "t2m_mean": [26.5],
            "t2m_max": [31.0],
            "t2m_min": [22.0],
            "precip_mm_sum": [15.0],
            "rh2m_mean": [81.0],
        }
    )
    df_features = pd.DataFrame(
        {
            "ubigeo": ["160101"],
            "departamento": ["LORETO"],
            "epi_year": [1999],  # fuera del rango con clima disponible
            "epi_week": [1],
            "cases_total": [10],
        }
    )

    result = add_climate_features(df_features, df_climate_weekly)

    assert len(result) == 1
    assert pd.isna(result.iloc[0]["t2m_mean"])
