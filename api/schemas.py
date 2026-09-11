"""Esquemas Pydantic de la API de inferencia (Fase 6, ver README)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HorizonPrediction(BaseModel):
    horizon: int = Field(description="Semanas hacia adelante (1-4).")
    champion: str = Field(
        description="Modelo campeon para este horizonte segun el ultimo "
        "entrenamiento (lightgbm_quantile, naive, seasonal_naive o moving_average_4)."
    )
    quantiles: dict[str, float] | None = Field(
        default=None,
        description="Prediccion por cuantil (q0.1/q0.5/q0.9) cuando el campeon "
        "es lightgbm_quantile. None cuando el campeon es un baseline (no da "
        "intervalo de incertidumbre, solo un punto).",
    )
    point_estimate: float | None = Field(
        default=None,
        description="Prediccion puntual cuando el campeon es un baseline "
        "(naive/seasonal_naive/moving_average_4). None cuando el campeon es "
        "lightgbm_quantile (usar 'quantiles' en su lugar; q0.5 es el punto central).",
    )


class UbigeoPrediction(BaseModel):
    ubigeo: str
    departamento: str
    provincia: str
    distrito: str
    as_of_epi_year: int = Field(description="Anio de la ultima semana con datos disponibles.")
    as_of_epi_week: int = Field(description="Semana epidemiologica de la ultima semana con datos.")
    predictions: list[HorizonPrediction]


class HealthResponse(BaseModel):
    status: str
    champion_config_loaded: bool
    n_ubigeos_available: int
