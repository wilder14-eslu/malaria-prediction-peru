# 🦟 Plataforma de Alerta Temprana de Malaria — Perú

**Sistema de predicción de casos de malaria por distrito y semana
epidemiológica**, construido con Machine Learning sobre datos reales del
Ministerio de Salud del Perú (2000–2024).

> Predice cuántos casos de malaria habrá en cada distrito del Perú en las
> próximas 1 a 4 semanas, con intervalos de confianza — permitiendo a las
> autoridades de salud priorizar recursos antes de que ocurran los brotes.

---

## 🎯 ¿Qué problema resuelve?

La malaria sigue siendo un problema de salud pública en la Amazonía peruana
(especialmente en Loreto, Amazonas y Madre de Dios). Las autoridades
sanitarias necesitan **anticiparse a los brotes** para:

- **Distribuir medicamentos e insumos** a los distritos que más los
  necesitarán en las próximas semanas.
- **Activar brigadas de fumigación y prevención** antes de que los casos se
  disparen, no después.
- **Optimizar presupuesto** focalizando recursos en los distritos con mayor
  riesgo proyectado, en vez de repartir uniformemente.
- **Reducir muertes y complicaciones** gracias a una respuesta más rápida y
  basada en datos.

Hoy, las decisiones de respuesta se toman **reactivamente**, cuando los
casos ya se reportaron. Esta plataforma permite pasar a un modelo
**predictivo**: saber hoy lo que probablemente pasará la semana que viene.

---

## 👥 ¿Para quién es?

| Stakeholder | Cómo lo usa |
|-------------|-------------|
| **DIRESA / GERESA** (Direcciones Regionales de Salud) | Consultan predicciones semanales por distrito para priorizar intervenciones y asignar personal de campo |
| **CDC Perú / DGE-MINSA** (Centro Nacional de Epidemiología) | Monitoreo nacional de riesgo, detección temprana de brotes inusuales, insumo para alertas epidemiológicas |
| **ONG y cooperación internacional** (OPS, USAID) | Planificación logística de donaciones de mosquiteros, pruebas rápidas y antimaláricos |
| **Investigadores en epidemiología** | Modelo reproducible para estudiar la dinámica espacio-temporal de la malaria en Perú |
| **Equipos de ciencia de datos en salud pública** | Arquitectura MLOps de referencia para construir sistemas similares en otras enfermedades (dengue, Zika, etc.) |

---

## 💡 Casos de uso concretos

### 1. Alerta semanal por distrito
> *"El distrito de Yurimaguas (UBIGEO 160601) tiene una predicción de 15
> casos (IC 80%: 5–28) para la semana 42. Esto es 3x más que su promedio
> histórico → activar alerta."*

### 2. Priorización de recursos logísticos
> *"De los 1,645 distritos monitoreados, 23 tienen predicción alta para las
> próximas 2 semanas. Concentrar el envío de pruebas rápidas y
> antimaláricos en estos 23 distritos."*

### 3. Evaluación de intervenciones
> *"Después de la campaña de fumigación en Iquitos, la predicción del modelo
> bajó un 40% en las semanas siguientes vs. el escenario sin intervención.
> La campaña tuvo impacto medible."*

### 4. Dashboard epidemiológico (integración futura)
> *La API está lista para ser consumida por un frontend/dashboard que
> muestre mapas de calor de riesgo por distrito, curvas de predicción vs.
> casos reales, y alertas automáticas.*

---

## 📊 Resultados del modelo

El modelo campeón (**LightGBM por cuantiles**) fue validado con evaluación
walk-forward sobre **143,502 registros UBIGEO × semana** (datos reales
completos, 1,645 distritos, 2000–2024):

| Horizonte | WAPE (campeón) | WAPE (mejor baseline) | Mejora |
|-----------|----------------|-----------------------|--------|
| 1 semana  | **0.456**      | 0.520 (naive)         | 12%    |
| 2 semanas | **0.506**      | 0.569 (media móvil)   | 11%    |
| 3 semanas | **0.528**      | 0.590 (media móvil)   | 10%    |
| 4 semanas | **0.542**      | 0.616 (media móvil)   | 12%    |

- **Cobertura del intervalo 80%** (q0.1–q0.9): entre **94.1% y 94.5%** →
  los intervalos de confianza son confiables para la toma de decisiones.
- Supera a los 3 baselines (naive, seasonal naive, media móvil) en los 4
  horizontes evaluados.

> **¿Qué significa WAPE 0.46?** Que el error promedio ponderado de las
> predicciones es ~46% del volumen real de casos. Para un distrito con 10
> casos reales, el modelo predice entre 5 y 15 en promedio — útil para
> planificación logística, no para diagnóstico clínico individual.

---

## 🚀 API de predicción

La API REST expone el modelo campeón para consultar predicciones en tiempo
real. Desplegada con **FastAPI** en **Render**.

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/predict/{ubigeo}` | Predicción de casos para un distrito (4 horizontes) |
| `GET` | `/health` | Estado del servicio y modelos cargados |
| `GET` | `/docs` | 📖 Documentación interactiva (Swagger UI) |
| `GET` | `/redoc` | 📖 Documentación alternativa (ReDoc) |

### Ejemplo real — Iquitos (UBIGEO 160101)

```bash
curl http://127.0.0.1:8000/predict/160101
```

```json
{
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
      "point_estimate": null
    },
    {
      "horizon": 2,
      "champion": "lightgbm_quantile",
      "quantiles": {"q0.1": 2.54, "q0.5": 9.12, "q0.9": 18.71},
      "point_estimate": null
    }
  ]
}
```

**Lectura del resultado:** Para Iquitos en la semana 1 siguiente, el modelo
predice una **mediana de ~11 casos** (q0.5), con un intervalo de confianza
80% entre **3 y 20 casos** (q0.1–q0.9).

### Documentación interactiva

La documentación de la API se genera automáticamente y está disponible en:
- **Swagger UI:** `{url_base}/docs` — interfaz interactiva para probar
  los endpoints directamente desde el navegador.
- **ReDoc:** `{url_base}/redoc` — documentación en formato legible.

> **Nota:** No existe un frontend/dashboard web por ahora. La API está
> preparada (CORS abierto) para que cualquier aplicación web, móvil o
> sistema externo la consuma.

---

## 🏗️ Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                    FUENTES DE DATOS                             │
│  📊 Datos Abiertos Perú (CDC/DGE-MINSA) — Vigilancia Malaria   │
│  🌡️ NASA POWER — Clima (temperatura, precipitación, humedad)   │
│  🗺️ INEI — Límites distritales y códigos UBIGEO                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 PIPELINE DE INGESTA                              │
│  • Unión de fuentes 2000-2008 + 2009-2024                       │
│  • Validación de esquemas (Pandera)                             │
│  • Limpieza de datos (UBIGEO vacíos, edades corruptas)          │
│  • Tabla canónica: UBIGEO × semana epidemiológica               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING                                │
│  📈 Temporal: lags, rolling means, tendencia, estacionalidad    │
│  🦠 Epidemiológico: ratio por especie, tasa de crecimiento      │
│  🗺️ Espacial: casos en distritos vecinos (adyacencia INEI)      │
│  🌡️ Clima: temperatura, precipitación, humedad (NASA POWER)     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            ENTRENAMIENTO Y EVALUACIÓN                           │
│  🏆 Campeón: LightGBM por cuantiles (q0.1, q0.5, q0.9)         │
│  📊 Evaluación walk-forward (sin data leakage)                  │
│  🔄 Champion/Challenger automático vía MLflow                   │
│  📉 3 baselines: naive, seasonal naive, media móvil              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│               API DE INFERENCIA                                 │
│  🚀 FastAPI → /predict/{ubigeo}, /health                        │
│  📖 Docs automáticas → /docs (Swagger), /redoc                  │
│  🐳 Docker → Render (free tier)                                 │
│  ⏱️ Cold start: ~25-30s (carga features de 1,645 distritos)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📈 Estado del proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| 0 | Scaffolding del proyecto | ✅ Hecho |
| 1 | Ingesta + validación de esquemas | ✅ Hecho |
| 2 | Feature engineering (temporal, epidemiológico, espacial) | ✅ Hecho |
| 3a | LightGBM cuantiles + features espaciales — **modelo campeón** | ✅ Validado con datos reales |
| 3b | Champion/Challenger + registro en MLflow | ✅ Hecho |
| 4 | Temporal Fusion Transformer + clima (NASA POWER) | ⚠️ Entrenado, pierde vs campeón (WAPE 1.33) |
| 5 | GNN espacio-temporal (opcional) | ⏸️ En pausa |
| 6 | API de inferencia (FastAPI) | ✅ Hecha y probada end-to-end |
| 7 | Monitoreo, drift y política de reentrenamiento | ⏸️ En pausa |
| 8 | Despliegue (Render) + CI/CD | 🔧 En progreso |

---

## 📂 Estructura del proyecto

```
malaria-prediction-peru/
├── api/                    # API de inferencia (FastAPI)
│   ├── main.py             #   Endpoints: /predict, /health, /docs
│   └── schemas.py          #   Modelos Pydantic de request/response
├── src/                    # Código de librería
│   ├── ingestion/          #   Carga y limpieza de datos crudos
│   ├── validation/         #   Esquemas Pandera
│   ├── features/           #   Feature engineering (temporal, epi, clima)
│   ├── preprocessing/      #   Transformaciones pre-modelo
│   ├── models/             #   LightGBM, TFT, baselines
│   ├── evaluation/         #   Métricas (WAPE, pinball loss, cobertura)
│   └── inference/          #   Lógica de predicción para la API
├── pipelines/              # Scripts orquestables
│   ├── ingestion/          #   run_ingestion.py
│   ├── training/           #   run_baselines, run_ml_models, persist_champion
│   └── reference/          #   Generación de tablas de referencia
├── configs/                # Configuración declarativa (YAML)
│   ├── data.yaml           #   Rutas de datos y fuentes
│   └── model.yaml          #   Hiperparámetros
├── models/                 # Artefactos del modelo campeón
│   ├── champion.json       #   Registro de campeón por horizonte
│   └── forecasting_h*_champion.joblib  # Modelos serializados
├── data/
│   ├── raw/                #   CSVs originales (no versionados, ~77 MB)
│   ├── gold/               #   Tabla canónica procesada (versionada)
│   └── reference/          #   Tablas auxiliares (adyacencia, nombres, etc.)
├── tests/                  # 45 tests (unitarios e integración)
├── docs/                   # Documentación de arquitectura
├── Dockerfile              # Imagen para despliegue en Render
├── render.yaml             # Configuración del servicio en Render
└── pyproject.toml          # Dependencias y configuración del proyecto
```

---

## 🔧 Guía de desarrollo

### Requisitos previos

- Python 3.12+
- Los dos CSVs de datos reales de la [Plataforma Nacional de Datos Abiertos](https://datosabiertos.gob.pe/dataset/vigilancia-epidemiol%C3%B3gica-de-malaria)
  (CDC Perú / DGE-MINSA) en `data/raw/epidemiology/`:
  - `datos_abiertos_vigilancia_malaria_2000_2008.csv` (~5.6 MB)
  - `datos_abiertos_vigilancia_malaria_2009_2024.csv` (~71 MB)

### Instalación

```bash
python -m venv .venv
source .venv/bin/activate  # o .venv\Scripts\activate en Windows
pip install -e ".[dev,monitoring]"
```

### Pipeline completo (de datos crudos a API funcionando)

```bash
# 1. Ingesta: CSV crudo → tabla canónica gold
python -m pipelines.ingestion.run_ingestion

# 2. Baselines (naive, seasonal naive, media móvil)
python -m pipelines.training.run_baselines

# 3. Modelo ML (LightGBM por cuantiles)
python -m pipelines.training.run_ml_models

# 4. Champion/Challenger + registro en MLflow
python -m pipelines.training.persist_champion_models

# 5. Levantar la API
uvicorn api.main:app --reload
# → Abrir http://127.0.0.1:8000/docs para la documentación interactiva
```

### Tests

```bash
pytest -v    # 45 tests, ~35 segundos
```

### Calidad de código

```bash
ruff check .          # Lint
ruff format .         # Formato
mypy src api          # Chequeo de tipos
```

### Tablas de referencia (solo la primera vez)

```bash
# Adyacencia distrital (requiere el extra "geo")
pip install -e ".[geo]"
python -m pipelines.reference.build_district_adjacency

# Nombres de distrito (no requiere "geo")
python -m pipelines.reference.build_district_names
```

### Etapa 2 — TFT + clima (experimental)

```bash
# Centroides departamentales
pip install -e ".[geo]"
python -m pipelines.reference.build_department_centroids

# Datos climáticos de NASA POWER (requiere internet, reanudable)
python pipelines/reference/fetch_climate_data.py

# Entrenar TFT (~20 min/epoch en CPU, reanudable)
pip install -e ".[dl]"
python -m pipelines.training.run_tft_model
```

> **Nota sobre TFT:** Entrenado con clima real de NASA POWER (25
> departamentos, 2000–2024). WAPE=1.33 — pierde contra los baselines y
> contra LightGBM. Probablemente por pocos epochs, sin búsqueda de
> hiperparámetros, y sin los lags que sí usa LightGBM. Ver
> `docs/architecture.md` para el análisis detallado.

---

## 🌐 Despliegue

El servicio se despliega en [Render](https://render.com/) (free tier) como
contenedor Docker:

```bash
# Build local para probar
docker build -t malaria-api .
docker run -p 8000:8000 malaria-api
```

La configuración de Render está en `render.yaml`:
- **Runtime:** Docker
- **Plan:** Free
- **Región:** Oregon
- **Health check:** `/health`

### Prerequisito para el build

Antes de construir la imagen Docker, el pipeline de entrenamiento debe
haberse corrido para generar los modelos en `models/`. Los artefactos
necesarios (`champion.json`, `forecasting_h*_champion.joblib`, `data/gold/`)
ya están versionados en git.

---

## 📝 Notas técnicas

### Datos

- **Fuente:** [Vigilancia Epidemiológica de Malaria](https://datosabiertos.gob.pe/dataset/vigilancia-epidemiol%C3%B3gica-de-malaria),
  CDC Perú / DGE-MINSA.
- **Cobertura:** 1,645 distritos, 2000–2024, 143,502 registros UBIGEO ×
  semana tras la unión de ambas fuentes.
- **Calidad:** 473 filas con nombres vacíos (auto-rellenados desde tabla
  INEI) y 44 filas con edad corrupta (limpiada a NaN). Manejado
  automáticamente en `src/ingestion/renace.py`.

### MLflow

La base de datos de MLflow se guarda en
`~/.mlflow-malaria-prediction-peru/` (fuera del repo) porque SQLite dentro
de una carpeta sincronizada con OneDrive puede fallar. Sobreescribible con
`MLFLOW_TRACKING_URI`.

### Limitaciones conocidas

- **Sin frontend/dashboard:** La API está lista (CORS abierto) para ser
  consumida por cualquier aplicación web, pero aún no existe un dashboard
  visual.
- **Features estáticas al arrancar:** Las features se calculan una sola vez
  al iniciar la API (~25-30s con datos completos). Para reflejar datos
  nuevos, hay que reiniciar el servicio después de re-correr la ingesta.
- **Sin monitoreo de drift:** No hay detección automática de degradación
  del modelo ni política de reentrenamiento.

---

## 📜 Licencia

MIT

## 👤 Contacto

**Wilder Eslu** — [github.com/wilder14-eslu](https://github.com/wilder14-eslu)
