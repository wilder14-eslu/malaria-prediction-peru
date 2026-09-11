"""Modelos de forecasting con salida por cuantiles (Etapa 1 del rediseno ML).

LightGBM soporta ``objective="quantile"`` nativamente, pero entrena un
modelo por cuantil (no hay forma nativa de sacar varios cuantiles de un
solo arbol de gradient boosting sin re-entrenar). Por eso
``train_quantile_models`` devuelve un modelo por cuantil solicitado.

LightGBM tambien maneja NaN en las features de forma nativa (las trata como
una categoria mas al decidir splits), asi que no hace falta imputar los NaN
que dejan los lags al inicio de cada serie -- solo hay que descartar filas
donde el TARGET es NaN (las ultimas ``h`` semanas de cada distrito, donde
el futuro todavia no se conoce).
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

# Any (no "object"): son kwargs que se pasan directo al constructor de
# LGBMRegressor, cuya firma tiene un tipo distinto por parametro
# (int, float, str, ...); "object" haria que mypy rechace el **-unpacking.
DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "num_leaves": 15,
    "learning_rate": 0.05,
    "min_child_samples": 10,
    "verbosity": -1,
}


def get_feature_columns(df: pd.DataFrame, horizons: tuple[int, ...]) -> list[str]:
    """Todas las columnas menos identificadores/geografia (no predictivas o
    con fuga de identidad) y las columnas de target (una por horizonte).
    """
    exclude = {
        "ubigeo",
        "departamento",
        "provincia",
        "distrito",
        "epi_year",
        "epi_week",
        "epi_index",
        *(f"target_h{h}" for h in horizons),
    }
    return [c for c in df.columns if c not in exclude]


def train_quantile_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    quantiles: tuple[float, ...],
    params: dict[str, Any] | None = None,
) -> dict[float, lgb.LGBMRegressor]:
    """Entrena un LightGBM por cuantil, descartando filas sin target valido."""
    valid = train_df.dropna(subset=[target_col])
    x_train = valid[feature_cols]
    y_train = valid[target_col]

    merged_params = {**DEFAULT_LGBM_PARAMS, **(params or {})}

    models = {}
    for q in quantiles:
        model = lgb.LGBMRegressor(objective="quantile", alpha=q, **merged_params)
        model.fit(x_train, y_train)
        models[q] = model
    return models


def predict_quantiles(
    models: dict[float, lgb.LGBMRegressor], df: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """Devuelve un DataFrame con una columna ``q{quantile}`` por cuantil
    entrenado, alineado al indice de ``df``.
    """
    x = df[feature_cols]
    predictions = {f"q{q}": models[q].predict(x) for q in models}
    result = pd.DataFrame(predictions, index=df.index)

    # Las predicciones de cuantiles distintos entrenados por separado pueden
    # cruzarse (ej. q0.1 > q0.5) porque cada uno es un modelo independiente.
    # Se ordenan fila a fila para garantizar q_bajo <= q_medio <= q_alto,
    # una practica estandar cuando no se usa un metodo de cuantiles conjunto.
    sorted_cols = sorted(result.columns, key=lambda c: float(c[1:]))
    result[sorted_cols] = np.sort(result[sorted_cols].to_numpy(), axis=1)
    return result
