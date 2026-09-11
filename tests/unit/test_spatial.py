"""Tests de features espaciales: suma de casos en distritos vecinos, con lag."""

from __future__ import annotations

import pandas as pd

from src.features.spatial import add_neighbor_lag_features


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_neighbor_cases_lag_sums_across_neighbors() -> None:
    # A y B son vecinos de C. En epi_index=10, A tenia 5 casos y B tenia 3.
    # C en epi_index=11 deberia ver neighbor_cases_lag_1 = 5 + 3 = 8.
    df = _df(
        [
            {"ubigeo": "A", "epi_index": 10, "cases_total": 5},
            {"ubigeo": "B", "epi_index": 10, "cases_total": 3},
            {"ubigeo": "C", "epi_index": 10, "cases_total": 0},
            {"ubigeo": "A", "epi_index": 11, "cases_total": 0},
            {"ubigeo": "B", "epi_index": 11, "cases_total": 0},
            {"ubigeo": "C", "epi_index": 11, "cases_total": 2},
        ]
    )
    adjacency = _df(
        [
            {"ubigeo": "C", "neighbor_ubigeo": "A"},
            {"ubigeo": "C", "neighbor_ubigeo": "B"},
        ]
    )

    result = add_neighbor_lag_features(df, adjacency, lag=1)

    row_c_11 = result.loc[(result["ubigeo"] == "C") & (result["epi_index"] == 11)].iloc[0]
    assert row_c_11["neighbor_cases_lag_1"] == 8
    assert row_c_11["neighbor_count"] == 2


def test_neighbor_cases_is_nan_when_district_has_no_neighbors_in_adjacency() -> None:
    # D no aparece en la tabla de adyacencia: no hay vecinos conocidos, no es "0".
    df = _df(
        [
            {"ubigeo": "D", "epi_index": 10, "cases_total": 0},
            {"ubigeo": "D", "epi_index": 11, "cases_total": 1},
        ]
    )
    adjacency = _df(
        {"ubigeo": pd.Series([], dtype=str), "neighbor_ubigeo": pd.Series([], dtype=str)}
    )

    result = add_neighbor_lag_features(df, adjacency, lag=1)

    row = result.loc[(result["ubigeo"] == "D") & (result["epi_index"] == 11)].iloc[0]
    assert pd.isna(row["neighbor_cases_lag_1"])


def test_neighbor_cases_uses_correct_lag_not_current_week() -> None:
    # Vecino unico E de F: en epi_index=10 tiene 100 casos, en epi_index=11
    # tiene 1 caso. F en epi_index=11 con lag=1 debe ver el valor de la
    # semana 10 (100), no el de la semana 11 (1) -- si no, hay fuga de futuro.
    df = _df(
        [
            {"ubigeo": "E", "epi_index": 10, "cases_total": 100},
            {"ubigeo": "E", "epi_index": 11, "cases_total": 1},
            {"ubigeo": "F", "epi_index": 10, "cases_total": 0},
            {"ubigeo": "F", "epi_index": 11, "cases_total": 0},
        ]
    )
    adjacency = _df([{"ubigeo": "F", "neighbor_ubigeo": "E"}])

    result = add_neighbor_lag_features(df, adjacency, lag=1)

    row_f_11 = result.loc[(result["ubigeo"] == "F") & (result["epi_index"] == 11)].iloc[0]
    assert row_f_11["neighbor_cases_lag_1"] == 100
