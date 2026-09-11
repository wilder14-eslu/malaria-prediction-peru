"""API de inferencia (Fase 6, ver README): expone el modelo campeon de cada
horizonte (entrenado por ``pipelines/training/persist_champion_models.py``)
para predecir casos de malaria por distrito.

Corre con: ``uvicorn api.main:app --reload`` (desarrollo) o
``uvicorn api.main:app --host 0.0.0.0 --port 8000`` (produccion, ver Fase 8
en README para despliegue en Render).

Limitacion conocida (ver docstring de ``src/inference/predict.py``): las
features se cargan una sola vez al arrancar la API (no hay actualizacion
incremental en caliente). Para que la API refleje datos nuevos hay que
reiniciarla despues de correr ``run_ingestion`` + `persist_champion_models``
de nuevo -- aceptable para esta primera version, a revisar en la Fase 7
(monitoreo/reentrenamiento) si hace falta servir con datos mas frescos sin
reiniciar.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from api.schemas import HealthResponse, HorizonPrediction, UbigeoPrediction
from src.inference.predict import UbigeoNotFoundError, build_latest_features, load_champion_config
from src.inference.predict import predict_for_ubigeo as _predict_for_ubigeo

logger = logging.getLogger("api")

DEFAULT_HORIZONS = (1, 2, 3, 4)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Carga features y config de campeones una vez al arrancar. Si algo
    falta (no se corrio el pipeline de entrenamiento todavia), la API igual
    levanta -- /health lo reporta y /predict devuelve 503 con un mensaje
    claro, en vez de que uvicorn falle al arrancar con un traceback críptico.
    """
    try:
        app.state.features = build_latest_features()
        app.state.champion_config = load_champion_config()
        app.state.ready = True
        logger.info("API lista: %d UBIGEO disponibles.", app.state.features["ubigeo"].nunique())
    except FileNotFoundError as exc:
        app.state.features = None
        app.state.champion_config = None
        app.state.ready = False
        logger.warning("API arranco sin datos/modelos listos: %s", exc)
    yield


app = FastAPI(
    title="Malaria Prediction Peru - API de inferencia",
    description=(
        "Prediccion de casos de malaria por distrito (UBIGEO) x semana "
        "epidemiologica, usando el modelo campeon de cada horizonte "
        "(ver docs/architecture.md)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ready = bool(getattr(app.state, "ready", False))
    n_ubigeos = int(app.state.features["ubigeo"].nunique()) if ready else 0
    return HealthResponse(
        status="ok" if ready else "not_ready",
        champion_config_loaded=ready,
        n_ubigeos_available=n_ubigeos,
    )


@app.get("/predict/{ubigeo}", response_model=UbigeoPrediction)
def predict(ubigeo: str) -> UbigeoPrediction:
    if not getattr(app.state, "ready", False):
        raise HTTPException(
            status_code=503,
            detail=(
                "El modelo todavia no esta entrenado/disponible. Corre "
                "'python -m pipelines.ingestion.run_ingestion' y "
                "'python -m pipelines.training.persist_champion_models', "
                "despues reinicia la API."
            ),
        )

    try:
        result: dict[str, Any] = _predict_for_ubigeo(
            ubigeo, app.state.features, app.state.champion_config, horizons=DEFAULT_HORIZONS
        )
    except UbigeoNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    predictions = [
        HorizonPrediction(horizon=h, **payload) for h, payload in result["predictions"].items()
    ]
    return UbigeoPrediction(
        ubigeo=result["ubigeo"],
        departamento=result["departamento"],
        provincia=result["provincia"],
        distrito=result["distrito"],
        as_of_epi_year=result["as_of_epi_year"],
        as_of_epi_week=result["as_of_epi_week"],
        predictions=predictions,
    )
