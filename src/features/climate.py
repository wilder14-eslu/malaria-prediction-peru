"""Carga y agregacion de datos climaticos (Etapa 2 del rediseno ML: Temporal
Fusion Transformer + clima, ver ``docs/architecture.md``).

Los datos crudos vienen de ``pipelines/reference/fetch_climate_data.py``
(NASA POWER, un punto por departamento -- ver ese script para la
justificacion de por que a nivel departamento y no distrito) como serie
DIARIA. Este modulo la agrega a semanal y la une a la tabla canonica de
casos, que esta a nivel UBIGEO x semana.

Aproximacion de semana epidemiologica: la fuente de vigilancia (RENACE) usa
su propia definicion de semana epidemiologica, que no esta documentada en
los metadatos del dataset abierto. Aca se usa la semana ISO-8601 (lunes a
domingo) como aproximacion -- puede desalinearse en 1 semana en los bordes
de anio con la semana RENACE real. Se documenta como limitacion conocida,
igual que ``WEEKS_PER_YEAR=52`` en ``src/features/temporal.py``.
"""

from __future__ import annotations

import pandas as pd


def aggregate_climate_to_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Agrega clima diario por departamento a nivel departamento x semana ISO.

    ``df_daily`` debe tener columnas ``departamento``, ``date`` (parseable a
    fecha) y las variables climaticas diarias (``t2m``, ``t2m_max``,
    ``t2m_min``, ``precip_mm``, ``rh2m``).

    La precipitacion se agrega por SUMA semanal (volumen total de lluvia,
    relevante para criaderos de mosquito); temperatura y humedad por
    PROMEDIO semanal.
    """
    df = df_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    iso = df["date"].dt.isocalendar()
    df["epi_year"] = iso["year"].astype(int)
    df["epi_week"] = iso["week"].astype(int)

    grouped = df.groupby(["departamento", "epi_year", "epi_week"], as_index=False).agg(
        t2m_mean=("t2m", "mean"),
        t2m_max=("t2m_max", "max"),
        t2m_min=("t2m_min", "min"),
        precip_mm_sum=("precip_mm", "sum"),
        rh2m_mean=("rh2m", "mean"),
    )
    return grouped


def add_climate_features(
    df_features: pd.DataFrame, df_climate_weekly: pd.DataFrame
) -> pd.DataFrame:
    """Une las features climaticas semanales por departamento a la tabla de
    features (a nivel UBIGEO x semana), por ``departamento`` x ``epi_year``
    x ``epi_week``. Left join: si una semana no tiene clima disponible (ej.
    fuera del rango descargado), las columnas climaticas quedan NaN en vez
    de perder filas de casos.
    """
    return df_features.merge(
        df_climate_weekly, on=["departamento", "epi_year", "epi_week"], how="left"
    )
