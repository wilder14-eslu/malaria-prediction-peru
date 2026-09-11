"""Orquesta el feature engineering completo sobre la tabla canonica:
grilla semanal completa -> lags -> rolling -> tendencia -> estacionalidad
-> ratios de especie.
"""

from __future__ import annotations

import pandas as pd

from src.features.epidemiological import add_species_ratio_features
from src.features.temporal import (
    add_cyclical_week_features,
    add_lag_features,
    add_rolling_features,
    add_trend_features,
    complete_weekly_grid,
)


def build_features(df_canonical: pd.DataFrame) -> pd.DataFrame:
    """Recibe la tabla canonica (una fila por UBIGEO x semana CON casos) y
    devuelve la tabla de features lista para entrenamiento/inferencia (una
    fila por UBIGEO x semana, incluyendo semanas sin casos).
    """
    df = complete_weekly_grid(df_canonical)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_trend_features(df)
    df = add_cyclical_week_features(df)
    df = add_species_ratio_features(df)
    return df
