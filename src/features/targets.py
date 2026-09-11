"""Creacion de targets multi-horizonte para forecasting (T+1..T+4).

``target_h{n}`` es el valor de ``target_col`` ``n`` semanas en el futuro,
por UBIGEO. Requiere que ``df`` ya tenga la grilla semanal completa
(``complete_weekly_grid``), para que "n semanas en el futuro" sea tiempo
real y no "la siguiente fila observada".
"""

from __future__ import annotations

import pandas as pd


def create_forecast_targets(
    df: pd.DataFrame,
    target_col: str = "cases_total",
    horizons: tuple[int, ...] = (1, 2, 3, 4),
) -> pd.DataFrame:
    df = df.sort_values(["ubigeo", "epi_index"]).copy()
    for h in horizons:
        df[f"target_h{h}"] = df.groupby("ubigeo")[target_col].shift(-h)
    return df
