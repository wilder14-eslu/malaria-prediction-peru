"""Features temporales: grilla semanal completa, lags, rolling stats, tendencia
y codificacion ciclica de la semana epidemiologica.

Importante: la tabla canonica solo tiene un registro por UBIGEO x semana
cuando hubo al menos un caso (viene de un ``groupby`` sobre casos reales).
Eso significa que, tal cual, un ``lag_1`` no seria "la semana calendario
anterior" sino "la semana anterior CON casos", lo cual arruina cualquier
feature de lag/rolling. Por eso ``complete_weekly_grid`` debe correr primero:
rellena con cases=0 las semanas sin casos, para que el tiempo transcurrido
entre filas sea real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Simplificacion: se asume 52 semanas epidemiologicas por anio para construir
# un indice temporal continuo. Los anios con 53 semanas (raros, dependen del
# calendario epidemiologico oficial) quedan con una semana "53" que no encaja
# perfectamente en este indice lineal; se revisara al integrar el calendario
# epidemiologico oficial completo con datos reales multi-anio.
WEEKS_PER_YEAR = 52

CASE_COLS = ["cases_falciparum", "cases_vivax", "cases_total"]
STATIC_COLS = ["departamento", "provincia", "distrito"]


def _epi_index(epi_year: pd.Series, epi_week: pd.Series) -> pd.Series:
    """Indice entero continuo y monotono para ordenar/rellenar semanas."""
    return epi_year * WEEKS_PER_YEAR + epi_week


def complete_weekly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena con cases=0 las semanas sin casos, por UBIGEO, entre su primera
    y ultima semana observada. Sin esto, los lags/rolling no representan
    tiempo real transcurrido.
    """
    df = df.copy()
    df["epi_index"] = _epi_index(df["epi_year"], df["epi_week"])

    filled_parts: list[pd.DataFrame] = []
    for ubigeo, group in df.groupby("ubigeo", sort=False):
        static_values = group[STATIC_COLS].iloc[0].to_dict()
        full_index = pd.RangeIndex(group["epi_index"].min(), group["epi_index"].max() + 1)

        full = pd.DataFrame({"epi_index": full_index})
        full["ubigeo"] = ubigeo
        for col, value in static_values.items():
            full[col] = value

        merged = full.merge(group[["epi_index", *CASE_COLS]], on="epi_index", how="left")
        merged[CASE_COLS] = merged[CASE_COLS].fillna(0).astype(int)
        filled_parts.append(merged)

    result = pd.concat(filled_parts, ignore_index=True)
    result["epi_year"] = result["epi_index"] // WEEKS_PER_YEAR
    result["epi_week"] = result["epi_index"] % WEEKS_PER_YEAR
    # epi_week == 0 significa que en realidad es la ultima semana del anio
    # anterior (ej. epi_index multiplo exacto de 52 -> semana 52, no semana 0).
    rollover = result["epi_week"] == 0
    result.loc[rollover, "epi_year"] = result.loc[rollover, "epi_year"] - 1
    result.loc[rollover, "epi_week"] = WEEKS_PER_YEAR

    ordered_cols = ["ubigeo", *STATIC_COLS, "epi_year", "epi_week", "epi_index", *CASE_COLS]
    return result[ordered_cols].sort_values(["ubigeo", "epi_index"]).reset_index(drop=True)


def add_lag_features(
    df: pd.DataFrame, target_col: str = "cases_total", lags: tuple[int, ...] = (1, 2, 3, 4)
) -> pd.DataFrame:
    """Agrega columnas ``lag_{n}`` del target, por UBIGEO, ordenado en el tiempo.

    Requiere que ``df`` ya haya pasado por ``complete_weekly_grid`` (semanas
    consecutivas sin huecos), si no los lags no representan tiempo real.
    """
    df = df.sort_values(["ubigeo", "epi_index"]).copy()
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("ubigeo")[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, target_col: str = "cases_total", window: int = 4
) -> pd.DataFrame:
    """Agrega media y desviacion estandar movil de las ``window`` semanas
    previas (sin incluir la semana actual, para no filtrar informacion futura).
    """
    df = df.sort_values(["ubigeo", "epi_index"]).copy()
    shifted = df.groupby("ubigeo")[target_col].shift(1)
    df[f"rolling_mean_{window}"] = shifted.groupby(df["ubigeo"]).transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    )
    df[f"rolling_std_{window}"] = shifted.groupby(df["ubigeo"]).transform(
        lambda s: s.rolling(window, min_periods=1).std()
    )
    return df


def add_trend_features(df: pd.DataFrame, target_col: str = "cases_total") -> pd.DataFrame:
    """Agrega ``growth_rate``: variacion porcentual respecto a la semana previa.

    Se protege division por cero (semana previa con 0 casos) devolviendo NaN
    en vez de +-inf, para que el modelo lo trate como "sin senal" en vez de
    un valor extremo artificial.
    """
    df = df.sort_values(["ubigeo", "epi_index"]).copy()
    previous = df.groupby("ubigeo")[target_col].shift(1)
    df["growth_rate"] = np.where(previous > 0, (df[target_col] - previous) / previous, np.nan)
    return df


def add_cyclical_week_features(df: pd.DataFrame, week_col: str = "epi_week") -> pd.DataFrame:
    """Codifica la semana epidemiologica como seno/coseno (estacionalidad
    ciclica: la semana 52 esta "cerca" de la semana 1 del anio siguiente).
    """
    df = df.copy()
    angle = 2 * np.pi * df[week_col] / WEEKS_PER_YEAR
    df["week_sin"] = np.sin(angle)
    df["week_cos"] = np.cos(angle)
    return df
