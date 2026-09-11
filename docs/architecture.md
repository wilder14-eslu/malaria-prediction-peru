# Arquitectura de la plataforma

> Este documento describe la arquitectura de la Plataforma de Alerta Temprana de Malaria para Perú y la fundamenta con literatura académica y patrones de arquitectura de industria para sistemas de MLOps aplicados a vigilancia epidemiológica.

## Fundamentación / Trabajos relacionados

El diseño de esta plataforma (ingesta de datos epidemiológicos → ingeniería de features → entrenamiento con selección de modelo campeón/challenger → inferencia batch y por API → monitoreo de drift y política de reentrenamiento) no es una elección arbitraria: sigue el patrón que reporta la literatura reciente sobre sistemas de alerta temprana basados en IA para vigilancia de enfermedades infecciosas, y en particular para enfermedades transmitidas por vectores (malaria, dengue).

**Sobre el enfoque general (forecasting de casos + clasificación de brote):**

Dos revisiones sistemáticas recientes confirman que combinar forecasting de series temporales de casos con un modelo de clasificación de riesgo de brote es el enfoque dominante en la literatura, no una construcción ad hoc de este proyecto:

- Wang et al., *Artificial intelligence in early warning systems for infectious disease surveillance: a systematic review*, Frontiers in Public Health / PMC, 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12230060/
- *Disease Outbreak Detection and Forecasting: A Review of Methods and Data Sources*, ACM Transactions on Computing for Healthcare (también disponible en arXiv), 2024. https://arxiv.org/html/2410.17290v1 — cataloga el espectro de modelos usado en la práctica: métodos estadísticos clásicos (ARIMA, SARIMA, Holt-Winters, CUSUM), ML clásico (XGBoost, Random Forest, SVM, Gaussian Process Regression) y deep learning (LSTM, Bi-LSTM, GRU, CNN), que corresponde directamente al espectro de candidatos que este proyecto evalúa (baselines, LightGBM, XGBoost).
- Colubri et al., *Early detection of disease outbreaks and non-outbreaks using incidence data: a framework using feature-based time series classification and machine learning*, PLOS Computational Biology, 2024. https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1012782

**Específico a malaria:**

- *Utilizing a novel high-resolution malaria dataset for climate-informed predictions with a deep learning transformer model*, PMC, 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC10754862/ — el antecedente más directamente comparable: predicción de casos de malaria usando variables climáticas, que respalda el uso de features epidemiológicas/temporales (estacionalidad, canal endémico, volatilidad).

**Específico a dengue en contextos de bajos recursos (comparable al contexto peruano):**

- *Bridging the predictive divide: A hybrid early warning system for scalable and real-time dengue surveillance in LMICs*, 2025. https://pubmed.ncbi.nlm.nih.gov/40754050/ — sistema híbrido diseñado explícitamente para países de ingresos bajos/medios, que respalda decisiones orientadas a infraestructura de bajo costo (por ejemplo, Render free tier en vez de un stack cloud completo).
- *Dengue Early Warning System and Outbreak Prediction Tool in Bangladesh Using Interpretable Tree-Based Machine Learning Model*, PMC, 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC12063067/ — respalda la elección de modelos basados en árboles (XGBoost/LightGBM) interpretables sobre cajas negras para el modelo de clasificación de brote.
- *A systematic review of dengue outbreak prediction models: current scenario and future directions*, PMC, 2022. https://pmc.ncbi.nlm.nih.gov/articles/PMC9956653/

**Sobre el patrón de arquitectura de producción (separación ingesta/entrenamiento/inferencia, registro de modelos campeón-challenger):**

- Hopsworks, *From MLOps to ML Systems with Feature/Training/Inference (FTI) Pipelines*. https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines — el patrón FTI (pipelines de Features, Training e Inference como componentes independientes y desacoplados) es la referencia de industria más citada para justificar separar la ingesta de datos, el entrenamiento de modelos y la inferencia en producción como módulos independientes, y es consistente con el patrón de registro de modelos Champion/Challenger vía MLflow y A/B testing en producción.

## Propuesta de rediseño: enfoque de modelado ML

A diferencia del proyecto original (commit `213df81`), el modelado se separa
en etapas incrementales, cada una validada contra la anterior antes de
avanzar. Esto sigue el patrón de "complejidad justificada por evidencia" que
recomiendan las revisiones citadas arriba: no tiene sentido saltar a deep
learning si un modelo mas simple ya captura la señal disponible.

**Etapa 1 -- LightGBM + features espaciales + salida por cuantiles.**
Un modelo LightGBM por cuantil (0.1/0.5/0.9) y por horizonte (1-4 semanas),
con features de lag/rolling/tendencia/estacionalidad ciclica y un feature
espacial (casos rezagados en distritos vecinos, via adyacencia real
calculada de los limites distritales de INEI). Selección Champion/Challenger
contra 3 baselines (naive, naive estacional, media móvil), con registro en
MLflow Model Registry.

Estado: **hecho y validado con los datos reales completos** del portal de
datos abiertos (143502 registros UBIGEO x semana). LightGBM le gana a los 3
baselines en los 4 horizontes (WAPE 0.46-0.54 vs. 0.52-1.15), con cobertura
del intervalo 80% entre 94.1% y 94.5% -- ver README para el detalle. Confirma
que hay señal real explotable con features tabulares antes de invertir en
arquitecturas mas caras.

**Etapa 2 -- Temporal Fusion Transformer + clima.**
Motivada directamente por Alonso Fernandez-Guerrero et al. (PMC10754862,
citado arriba): un TFT que aprende su propia representación temporal por
distrito (en vez de un modelo por horizonte) y que puede usar clima
(temperatura, precipitación, humedad) como covariable -- variables con
vínculo biológico directo a la transmisión de malaria (ciclo de vida del
vector, criaderos) que Etapa 1 no usa.

Decisión de escala geográfica del clima: no existe una fuente pública
gratuita de clima ya agregada a nivel distrital para el Perú, y consultar
clima por los ~1800 distritos vía API sería lento y probablemente bloqueado
por rate limiting. Se usa **clima a nivel departamento** (25, no ~1800),
desde la API pública y gratuita de NASA POWER en el centroide de cada
departamento. Es una simplificación real -- pierde variación climática
dentro de un departamento grande como Loreto -- documentada aca en vez de
asumida en silencio; si en el futuro se justifica el costo, se puede
refinar a nivel provincia o con datos satelitales agregados por polígono
(ERA5/CHIRPS), que requieren mucho más procesamiento (NetCDF de varios GB).

Estado: **entrenado y validado con clima real** (25 departamentos, NASA
POWER, 2000-2024). Entrenamiento con early stopping (`patience=5`):
paró en el epoch 8 porque el epoch 2 (el mejor, `val_loss=0.2347`) no se
superó en los 5 epochs siguientes. En el set de validación: **WAPE=1.33,
MAE=0.32** -- peor que los 3 baselines (WAPE 0.52-1.15) y muy por debajo del
campeón de Etapa 1 (LightGBM, WAPE 0.46-0.54). **No reemplaza al campeón
actual**; la API de inferencia (Fase 6) sigue sirviendo LightGBM.

Un bug real encontrado en el camino, ya corregido (ver `train_tft` en
`src/models/tft.py` y su test de regresión): con early stopping, el modelo
que queda en memoria al terminar `trainer.fit()` es el del ÚLTIMO epoch
entrenado, no el de mejor `val_loss` -- evaluar ese último epoch daba
WAPE=3.34 (evidentemente degradado), no WAPE=1.33. `train_tft` ahora
recarga los pesos del mejor checkpoint antes de devolver el modelo.

Por qué probablemente pierde contra Etapa 1 tal como está configurado, y
qué se necesitaría para darle una oportunidad más justa (no probado todavía
por el costo de cómputo -- ver nota de timing real más abajo y en
`run_tft_model.py`):
- Solo 8 epochs de entrenamiento real (~140 minutos en un sandbox de 2 CPU
  sin GPU, a ~20 min/epoch): un TFT con embeddings por serie para ~1637
  distritos probablemente necesita bastantes más epochs que un modelo
  tabular como LightGBM para converger.
- Hiperparámetros default sin ajustar (`learning_rate=0.03`,
  `hidden_size=16`): no se corrió ninguna búsqueda de hiperparámetros,
  a diferencia de Etapa 1.
- El TFT no recibe los `lag_*`/`rolling_*` que sí usa LightGBM (ver
  docstring de `src/models/tft.py`) -- por diseño, para que aprenda su
  propia representación, pero eso también le saca una ventaja de partida
  frente a Etapa 1 en un dataset relativamente chico.
- 214 de los ~1851 UBIGEO no entran al dataset (historia insuficiente para
  encoder+prediction window) y 2 más se excluyen del set de validación
  (ver `filter_known_groups` -- su primer caso cae entre el corte de train
  y el de validación), reduciendo aún más los datos por serie.

Entrenar esto en el sandbox de este proyecto (2 CPU, sin GPU) es lento:
~20 min/epoch reales medidos, y el entorno se suspende durante huecos de
inactividad entre interacciones (no es un problema del código, es una
restricción del entorno de ejecución en la nube) -- por eso
`train_tft`/`run_tft_model.py` soportan entrenamiento resumible por epoch
(`checkpoint_dir`/`resume_from_checkpoint`, ver docstring). En una máquina
con GPU o más núcleos, correr más epochs y una búsqueda de hiperparámetros
básica es mucho más barato y sería el primer paso antes de descartar
Etapa 2 definitivamente.

**Etapa 3 (opcional) -- GNN espacio-temporal.**
Si Etapa 2 mejora sobre Etapa 1 y se justifica la complejidad adicional: un
grafo donde cada nodo es un distrito (usando la misma adyacencia real que ya
alimenta el feature espacial de Etapa 1) y aristas ponderadas por
adyacencia/distancia, con una capa de atención espacio-temporal (ej.
DengueGNN, redes de atención de grafos espacio-temporales) para modelar
contagio entre distritos vecinos de forma mas rica que un simple lag de
vecinos. No iniciada -- depende del resultado de Etapa 2.

## Lo que sigue

- Etapa 2 (TFT) quedó por debajo de Etapa 1 con la config actual (ver
  arriba) -- si se quiere darle una oportunidad más justa: más epochs +
  una búsqueda básica de hiperparámetros (learning rate, hidden_size),
  idealmente en una máquina con GPU o más núcleos que este sandbox. No es
  la prioridad inmediata: LightGBM (Etapa 1) ya es el campeón real,
  validado, y sirviendo en la API (Fase 6).
- Etapa 3 (GNN) queda en espera -- no tiene sentido invertir en más
  complejidad sobre una Etapa 2 que todavía no supera a Etapa 1.
- Diagrama de componentes / flujo de datos end-to-end y stack de despliegue
  (Render free tier) -- pendiente, se agrega cuando la Fase 8 (despliegue)
  esté más cerca.
