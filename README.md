# Plataforma de Alerta Temprana de Malaria - Peru

Plataforma MLOps end-to-end para predecir casos de malaria y riesgo de brote
por distrito (UBIGEO) x semana epidemiologica en Peru.

Este repo se reconstruyo desde cero a partir del commit inicial
(`213df81 Add files via upload`), con un rediseno del enfoque de modelado ML.
Ver `docs/architecture.md` para el detalle completo del diseno y su
fundamentacion en literatura academica.

## Estado actual

| Fase | Descripcion | Estado |
|------|-------------|--------|
| 0 | Scaffolding del proyecto | Hecho |
| 1 | Ingesta + validacion de esquemas | Hecho |
| 2 | Feature engineering (temporal, epidemiologico, espacial) | Hecho |
| 3 | Etapa 1 del rediseno ML: LightGBM + features espaciales + salida por cuantiles | Hecho, validado con datos reales completos |
| 3b | Champion/Challenger + registro en MLflow | Hecho |
| 4 | Etapa 2 del rediseno ML: Temporal Fusion Transformer + clima | Entrenado y validado con clima real; WAPE 1.33, pierde contra Etapa 1 (ver nota abajo) |
| 5 | Etapa 3 del rediseno ML (opcional): GNN espacio-temporal | Pendiente |
| 6 | API de inferencia (FastAPI) | Hecho, probada de punta a punta contra los modelos campeon reales |
| 7 | Monitoreo, drift y politica de reentrenamiento | Pendiente |
| 8 | Despliegue (Render) + CI/CD | Pendiente |

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate  # o .venv\Scripts\activate en Windows
pip install -e ".[dev,monitoring]"
```

## Comandos principales

```bash
ruff check .          # lint
ruff format .         # formato
mypy src api          # tipos
pytest -v             # tests
python -m pipelines.ingestion.run_ingestion   # correr la ingesta
python -m pipelines.training.run_baselines    # baselines (naive, seasonal naive, media movil)
python -m pipelines.training.run_ml_models    # LightGBM por cuantiles (Etapa 1 del rediseno)
python -m pipelines.training.persist_champion_models  # Champion/Challenger + registro en MLflow
uvicorn api.main:app --reload   # API de inferencia (Fase 6), ver nota abajo

# Solo la primera vez (o para regenerar la adyacencia distrital):
pip install -e ".[geo]"
python -m pipelines.reference.build_district_adjacency

# Solo la primera vez (o para regenerar los nombres de distrito; no necesita
# el extra "geo", solo lee las propiedades del GeoJSON, no la geometria):
python -m pipelines.reference.build_district_names

# Etapa 2 (TFT + clima) -- ver nota de clima mas abajo antes de correr esto:
pip install -e ".[dl]"
python -m pipelines.reference.build_department_centroids   # necesita el extra "geo"
python pipelines/reference/fetch_climate_data.py            # necesita internet, correrlo aparte
python -m pipelines.training.run_tft_model
```

**Nota sobre los datos:** `data/raw/epidemiology/` debe contener los dos CSV
reales de la Plataforma Nacional de Datos Abiertos (dataset "Vigilancia
epidemiologica de Malaria", CDC Peru / DGE-MINSA):
https://datosabiertos.gob.pe/dataset/vigilancia-epidemiol%C3%B3gica-de-malaria
(`datos_abiertos_vigilancia_malaria_2000_2008.csv`, ~5.6 MB, y
`datos_abiertos_vigilancia_malaria_2009_2024.csv`, ~71 MB). Los nombres de
archivo coinciden exactamente con lo que espera `configs/data.yaml`, no hace
falta renombrar nada. No se versionan en git (`data/raw/` esta en
`.gitignore`), asi que hay que descargarlos/copiarlos localmente.

La fuente real trae dos problemas de calidad de datos menores, ya
manejados en `src/ingestion/renace.py` (no hace falta limpiarlos a mano):
- 473 de 106355 filas de la fuente 2000-2008 traen el UBIGEO correcto pero
  departamento/provincia/distrito vacios (los UBIGEO 160109 y 160114, en
  Loreto). Se rellenan automaticamente desde
  `data/reference/district_names.csv` (tabla de referencia INEI, generada
  con `pipelines/reference/build_district_names.py`) en vez de descartar
  esos casos reales de malaria.
- 44 de 597142 filas de la fuente 2009-2024 traen un valor de `edad`
  corrupto (ej. `90130297`, casi seguro otro campo pegado por error en la
  exportacion). Se limpian a NaN en vez de descartar la fila completa,
  porque `edad` no se usa en el modelado (ver nota en
  `src/features/epidemiological.py`).

**Resultado con los datos reales completos** (143502 registros UBIGEO x
semana tras la union de ambas fuentes, split walk-forward hasta 2024):
LightGBM por cuantiles le gana a los 3 baselines en los 4 horizontes
evaluados (WAPE 0.46-0.54 contra 0.52-1.15 de los baselines), con
cobertura del intervalo 80% (q0.1-q0.9) entre 94.1% y 94.5%. Es senal
real, no el artefacto de la muestra chica -- valida la Etapa 1 del
rediseno ML descrito en `docs/architecture.md`.

**Nota sobre clima y resultado real (Etapa 2, TFT):** entrenado con clima
real de NASA POWER (25 departamentos, 2000-2024) contra los datos reales
completos. Resultado en validacion: **WAPE=1.33, MAE=0.32** -- pierde
contra los 3 baselines (WAPE 0.52-1.15) y contra el campeon de Etapa 1
(LightGBM, WAPE 0.46-0.54). **La API sigue usando LightGBM como campeon**;
el TFT no lo reemplaza. Ver `docs/architecture.md` para el detalle de por
que probablemente pierde (pocos epochs por costo de computo, sin busqueda
de hiperparametros, sin los lags que si usa LightGBM) y que se necesitaria
para darle una oportunidad mas justa.

Para reproducir el entrenamiento (o intentar mejorarlo):

1. `pip install -e ".[geo]"` (si no esta ya) y
   `python -m pipelines.reference.build_department_centroids` -- calcula el
   centroide de cada uno de los 25 departamentos (no hay forma practica de
   pedir clima por los ~1800 distritos, seria demasiado lento/bloqueado por
   rate limiting; es una simplificacion real, documentada en
   `docs/architecture.md`).
2. `python pipelines/reference/fetch_climate_data.py` -- descarga clima
   diario 2000-2024 (temperatura, precipitacion, humedad) desde la API
   publica y gratuita de NASA POWER (https://power.larc.nasa.gov/, sin API
   key), un punto por departamento. Se corre en tu maquina (necesita
   internet), es reanudable si se corta, y solo usa la libreria estandar de
   Python.
3. `pip install -e ".[dl]"` (pytorch + pytorch-forecasting + lightning,
   ~2GB) y `python -m pipelines.training.run_tft_model`. En una maquina sin
   GPU, cada epoch tarda ~20 minutos con los datos reales completos (medido
   en un sandbox de 2 CPU); el entrenamiento es resumible por epoch si se
   corta a mitad de camino (ver docstring de `train_tft` en
   `src/models/tft.py`), y vuelve a arrancar solo con el mismo comando.

**Nota sobre la API de inferencia (Fase 6):** expone el modelo campeon de
cada horizonte (`GET /predict/{ubigeo}`, `GET /health`; docs interactivas
en `/docs` una vez levantada). Probada de punta a punta contra los modelos
reales entrenados con los datos completos -- ejemplo real, distrito de
Iquitos (UBIGEO 160101):

```bash
$ curl http://127.0.0.1:8000/predict/160101
{
  "ubigeo": "160101", "departamento": "LORETO", "provincia": "MAYNAS", "distrito": "IQUITOS",
  "as_of_epi_year": 2024, "as_of_epi_week": 52,
  "predictions": [
    {"horizon": 1, "champion": "lightgbm_quantile",
     "quantiles": {"q0.1": 3.28, "q0.5": 10.80, "q0.9": 20.43}, "point_estimate": null},
    ...
  ]
}
```

Limitacion conocida: las features se recalculan en memoria al arrancar la
API (no en cada request), no hay un feature store con actualizacion
incremental. Medido con los datos reales completos (1645 UBIGEO): arranque
en ~25-30 segundos; ver el detalle y por que en el docstring de
`src/inference/predict.py`. Si el pipeline de entrenamiento todavia no
corrio (falta `models/champion.json` o la tabla canonica), la API igual
levanta -- `/health` reporta `"status": "not_ready"` y `/predict` devuelve
503 con un mensaje claro, en vez de fallar el arranque.

**Nota sobre MLflow:** `persist_champion_models` guarda su base de datos
(necesaria para el Model Registry) en `~/.mlflow-malaria-prediction-peru/`,
fuera de esta carpeta, a proposito: como este repo esta en OneDrive, SQLite
dentro de una carpeta sincronizada en la nube puede fallar con errores de
E/S por como OneDrive bloquea archivos (se verifico este problema al
construir el pipeline). Sobreescribible con la variable de entorno
`MLFLOW_TRACKING_URI` si prefieres otra ubicacion.

## Estructura

```
src/            # codigo de libreria (ingesta, validacion, features, modelos, inferencia)
api/            # API de inferencia (FastAPI, Fase 6): esquemas + endpoints
pipelines/      # scripts orquestables (ingesta, entrenamiento, prediccion, monitoreo)
configs/        # configuracion declarativa (rutas de datos, hiperparametros, monitoreo)
tests/          # tests unitarios e de integracion
data/           # datos crudos y generados (no versionados, ver .gitignore)
docs/           # documentacion de arquitectura y decisiones de diseno
```

Los dos notebooks (`.ipynb`) en la raiz del repo son del commit inicial
(`213df81 Add files via upload`) y no forman parte de este rediseno; no se
modifican.
