# ---------------------------------------------------------------------------
# Dockerfile — Malaria Prediction Peru (API de inferencia)
#
# Construye una imagen lista para desplegar en Render (o cualquier plataforma
# que soporte contenedores Docker). Solo incluye las dependencias de
# produccion (sin dev/dl/geo) y los artefactos que la API necesita para
# arrancar: modelos campeon (.joblib + champion.json), datos de referencia
# y la tabla canonica gold.
#
# Prerequisito: antes de construir la imagen, el pipeline de entrenamiento
# debe haberse corrido localmente para generar los artefactos en models/:
#   python -m pipelines.ingestion.run_ingestion
#   python -m pipelines.training.run_ml_models
#   python -m pipelines.training.persist_champion_models
#
# Build:   docker build -t malaria-api .
# Run:     docker run -p 8000:8000 malaria-api
# ---------------------------------------------------------------------------

FROM python:3.12-slim AS base

# Evitar prompts interactivos y bytecode innecesario
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- Instalar dependencias del sistema (LightGBM necesita libgomp) ---------
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# --- Dependencias de Python ------------------------------------------------
# Copiar solo pyproject.toml primero para aprovechar la cache de Docker
COPY pyproject.toml ./
RUN pip install --no-cache-dir . && \
    pip cache purge 2>/dev/null || true

# --- Codigo fuente ---------------------------------------------------------
COPY src/ src/
COPY api/ api/
COPY pipelines/ pipelines/
COPY configs/ configs/

# --- Artefactos de modelos y datos -----------------------------------------
# Modelos campeon (generados por persist_champion_models)
COPY models/champion.json models/champion.json
COPY models/forecasting_h*_champion.joblib models/

# Datos de referencia (versionados en git)
COPY data/reference/ data/reference/

# Datos gold: tabla canonica + features materializadas (la API carga el
# parquet materializado directamente, sin recalcular, para caber en 512MB)
COPY data/gold/ data/gold/

# --- Runtime ---------------------------------------------------------------
EXPOSE 8000

# Render inyecta $PORT; si no existe, default a 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
