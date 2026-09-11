"""Modelos baseline multi-horizonte (T+1..T+4): naive, seasonal naive y
media movil. Sirven como piso de comparacion: si LightGBM/XGBoost no le
gana a estos, no vale la pena su complejidad adicional.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

WEEKS_PER_YEAR = 52


def predict_naive(
    df: pd.DataFrame, horizons: tuple[int, ...] = (1, 2, 3, 4), target_col: str = "cases_total"
) -> dict[int, pd.Series]:
    """El pronostico para cualquier horizonte es el ultimo valor observado."""
    return {h: df[target_col] for h in horizons}


def predict_seasonal_naive(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 2, 3, 4),
    target_col: str = "cases_total",
    weeks_per_year: int = WEEKS_PER_YEAR,
) -> dict[int, pd.Series]:
    """El pronostico para el horizonte h es el valor observado en la misma
    semana del anio anterior (ajustado por h semanas). Requiere que ``df``
    tenga la grilla semanal completa (``epi_index`` continuo).
    """
    df_sorted = df.sort_values(["ubigeo", "epi_index"])
    preds = {}
    for h in horizons:
        shift_n = weeks_per_year - h
        preds[h] = df_sorted.groupby("ubigeo")[target_col].shift(shift_n)
    return preds


def predict_moving_average(
    df: pd.DataFrame, horizons: tuple[int, ...] = (1, 2, 3, 4), window: int = 4
) -> dict[int, pd.Series]:
    """El pronostico para cualquier horizonte es la media movil de las
    ultimas ``window`` semanas (columna ``rolling_mean_{window}``, ya
    calculada por ``src/features/temporal.py`` -- debe correr antes).
    """
    col = f"rolling_mean_{window}"
    return {h: df[col] for h in horizons}


# Los tres predictores tienen firmas distintas (target_col vs window), pero
# todos aceptan (df, horizons=...) por keyword: se anota como Callable
# generico para que mypy no exija una firma identica. Centralizado aca (en
# vez de duplicado en cada pipeline que lo usa: run_baselines.py,
# persist_champion_models.py, src/inference/predict.py) para que agregar un
# baseline nuevo sea un solo lugar que tocar.
BaselinePredictor = Callable[..., dict[int, pd.Series]]

BASELINE_PREDICTORS: dict[str, BaselinePredictor] = {
    "naive": predict_naive,
    "seasonal_naive": predict_seasonal_naive,
    "moving_average_4": predict_moving_average,
}
