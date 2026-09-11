"""Metricas de evaluacion para forecasting: puntuales (MAE, RMSE, WAPE) y
de intervalo/cuantil (pinball loss, cobertura), estas ultimas necesarias
para la Etapa 1 del rediseno ML (salida por cuantiles en vez de un solo
punto; ver docs/architecture.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _aligned_valid_pairs(y_true: pd.Series, y_pred: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Alinea por indice y descarta filas con NaN en cualquiera de los dos
    (ej. las primeras semanas de la serie, que no tienen lags completos).
    """
    aligned = pd.concat([y_true, y_pred], axis=1, keys=["y_true", "y_pred"]).dropna()
    return aligned["y_true"].to_numpy(), aligned["y_pred"].to_numpy()


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    yt, yp = _aligned_valid_pairs(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    yt, yp = _aligned_valid_pairs(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def wape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Weighted Absolute Percentage Error: mas robusto que MAPE cuando hay
    semanas con 0 casos (no explota por division entre cero fila a fila).
    """
    yt, yp = _aligned_valid_pairs(y_true, y_pred)
    denom = np.sum(np.abs(yt))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(yt - yp)) / denom)


def pinball_loss(y_true: pd.Series, y_pred: pd.Series, quantile: float) -> float:
    """Perdida de cuantil (quantile/pinball loss): penaliza mas fuerte el
    lado del error que corresponde a estar del lado equivocado del cuantil.
    Es la metrica que se optimiza al entrenar con ``objective="quantile"``.
    """
    yt, yp = _aligned_valid_pairs(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    diff = yt - yp
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def interval_coverage(y_true: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    """Fraccion de observaciones que cayeron dentro del intervalo
    [lower, upper]. Para un intervalo del 80% (cuantiles 0.1/0.9) bien
    calibrado, deberia acercarse a 0.80.
    """
    aligned = pd.concat([y_true, lower, upper], axis=1, keys=["y_true", "lower", "upper"]).dropna()
    if len(aligned) == 0:
        return float("nan")
    within = (aligned["y_true"] >= aligned["lower"]) & (aligned["y_true"] <= aligned["upper"])
    return float(within.mean())
