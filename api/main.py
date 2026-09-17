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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html

from api.schemas import HealthResponse, HorizonPrediction, UbigeoPrediction
from src.inference.predict import UbigeoNotFoundError, build_latest_features, load_champion_config
from src.inference.predict import predict_for_ubigeo as _predict_for_ubigeo

logger = logging.getLogger("api")

DEFAULT_HORIZONS = (1, 2, 3, 4)

API_DESCRIPTION = """
## 🦟 Plataforma de Alerta Temprana de Malaria — Perú

API de inferencia para predecir **casos de malaria por distrito (UBIGEO)
y semana epidemiológica**, usando modelos de Machine Learning entrenados
con datos reales de la [Plataforma Nacional de Datos Abiertos](https://datosabiertos.gob.pe/dataset/vigilancia-epidemiol%C3%B3gica-de-malaria)
(CDC Perú / DGE-MINSA, 2000–2024).

### 🎯 Modelo campeón

**LightGBM por cuantiles** (Etapa 1), validado con walk-forward split sobre
143,502 registros UBIGEO × semana:

| Métrica | Valor |
|---------|-------|
| WAPE (h1–h4) | 0.46 – 0.54 |
| Cobertura intervalo 80% | 94.1% – 94.5% |
| Baselines superados | naive, seasonal naive, media móvil |

### 📖 Uso rápido

```bash
# Estado del servicio
curl https://tu-dominio.onrender.com/health

# Predicción para Iquitos (UBIGEO 160101)
curl https://tu-dominio.onrender.com/predict/160101
```

### 📊 Respuesta de predicción

Cada predicción incluye **4 horizontes** (1–4 semanas adelante) con:
- **q0.1** → límite inferior (cuantil 10%)
- **q0.5** → estimación central (mediana)
- **q0.9** → límite superior (cuantil 90%)

### 🗂️ Datos de entrada

La API cubre **1,645 distritos** con historia de vigilancia epidemiológica.
El código UBIGEO sigue el estándar INEI de 6 dígitos (ej. `160101` = Iquitos,
Maynas, Loreto).

---

*Repositorio: [github.com/wilder14-eslu/malaria-prediction-peru](https://github.com/wilder14-eslu/malaria-prediction-peru)*
"""

tags_metadata = [
    {
        "name": "Predicción",
        "description": "Genera predicciones de casos de malaria por distrito y horizonte temporal.",
    },
    {
        "name": "Monitoreo",
        "description": "Verifica el estado del servicio y la disponibilidad de modelos.",
    },
]


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
    title="🦟 Malaria Prediction Peru — API",
    description=API_DESCRIPTION,
    version="0.1.0",
    lifespan=lifespan,
    openapi_tags=tags_metadata,
    contact={
        "name": "Wilder Eslu",
        "url": "https://github.com/wilder14-eslu",
    },
    license_info={
        "name": "MIT",
    },
)

# Permitir que cualquier frontend (o Swagger UI externo) consuma la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoreo"],
    summary="Estado del servicio",
    description="Verifica si la API está lista y cuántos distritos (UBIGEO) tiene cargados.",
    responses={
        200: {
            "description": "Estado actual del servicio.",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "champion_config_loaded": True,
                        "n_ubigeos_available": 1645,
                    }
                }
            },
        }
    },
)
def health() -> HealthResponse:
    ready = bool(getattr(app.state, "ready", False))
    n_ubigeos = int(app.state.features["ubigeo"].nunique()) if ready else 0
    return HealthResponse(
        status="ok" if ready else "not_ready",
        champion_config_loaded=ready,
        n_ubigeos_available=n_ubigeos,
    )


@app.get(
    "/predict/{ubigeo}",
    response_model=UbigeoPrediction,
    tags=["Predicción"],
    summary="Predecir casos de malaria para un distrito",
    description=(
        "Dado un código UBIGEO (6 dígitos, estándar INEI), devuelve la predicción "
        "de casos de malaria para los próximos 1–4 horizontes semanales, usando el "
        "modelo campeón (LightGBM por cuantiles). Incluye intervalos de incertidumbre "
        "(q0.1, q0.5, q0.9)."
    ),
    responses={
        200: {
            "description": "Predicción exitosa.",
            "content": {
                "application/json": {
                    "example": {
                        "ubigeo": "160101",
                        "departamento": "LORETO",
                        "provincia": "MAYNAS",
                        "distrito": "IQUITOS",
                        "as_of_epi_year": 2024,
                        "as_of_epi_week": 52,
                        "predictions": [
                            {
                                "horizon": 1,
                                "champion": "lightgbm_quantile",
                                "quantiles": {"q0.1": 3.28, "q0.5": 10.80, "q0.9": 20.43},
                                "point_estimate": None,
                            },
                        ],
                    }
                }
            },
        },
        404: {"description": "UBIGEO no encontrado en la fuente de vigilancia."},
        503: {"description": "Modelo no disponible — el pipeline de entrenamiento no se ha corrido."},
    },
)
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
