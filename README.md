# 🦟 Predicción de Brotes de Malaria en el Perú para el 2027
## Mediante uso de Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.x-189AB4?style=for-the-badge)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![UNMSM](https://img.shields.io/badge/UNMSM-E.P.%20Estadística-8B0000?style=for-the-badge)
![Estado](https://img.shields.io/badge/Estado-En%20Progreso-yellow?style=for-the-badge)

</div>

---

## 📋 Descripción

Este proyecto de investigación desarrolla y compara múltiples modelos de **Machine Learning** para predecir brotes de malaria en regiones geográficas del Perú hasta el año **2027**, utilizando datos históricos de vigilancia epidemiológica del período **2000–2022** (604,968 registros).

La investigación contribuye al **"Plan hacia la eliminación de Malaria en el Perú 2022–2030"** del Ministerio de Salud, apoyando la identificación de áreas de alto riesgo para optimizar la asignación de recursos sanitarios.

Presentado en **EXPOANDINA** — *"La estadística como soporte a la toma de decisiones"*

---

## 👨‍🔬 Autores

| Autor | ORCID | Institución |
|-------|-------|-------------|
| **Espinoza Luna, Wilder Gilmer** | [0000-0002-0306-9789](https://orcid.org/0000-0002-0306-9789) | E.P. Estadística – UNMSM |
| **Arapa Lazo, Julio Cesar** | [0000-0001-9658-8379](https://orcid.org/0000-0001-9658-8379) | E.P. Estadística – UNMSM |

**Universidad Nacional Mayor de San Marcos** — Decana de América  
Escuela Profesional de Estadística · Área de Ciencias Matemáticas · Lima, Perú · 2024

---

## 🎯 Objetivos

### Objetivo General
Desarrollar un modelo de predicción preciso y confiable que determine las **regiones geográficas con mayor riesgo de incidencia de malaria al 2027**.

### Objetivos Específicos

| # | Dimensión | Variables | Pregunta de investigación |
|---|-----------|-----------|---------------------------|
| 1 | 🗺️ **Geográfica** | `departamento`, `provincia`, `distrito`, `localidad`, `ubigeo`, `localcod` | ¿Es posible predecir brotes de malaria por región? |
| 2 | 🔬 **Diagnóstica** | `enfermedad`, `diagnostic` | ¿Es posible predecir los tipos de malaria que más afectan a cada región? |
| 3 | 👥 **Demográfica** | `sexo`, `edad`, `tipo_edad` | ¿Es posible predecir grupos de población vulnerables? |
| 4 | 📅 **Temporal** | `semana`, `ano` | ¿Es posible predecir épocas del año con mayor prevalencia? |

---

## 📊 Dataset

| Atributo | Tipo | Descripción | Valores |
|---|---|---|---|
| `departamento` | Caracter | Región geográfica | — |
| `provincia` | Caracter | Provincia | — |
| `distrito` | Caracter | Lugar probable de infección | — |
| `localidad` | Caracter | Localidad | — |
| `enfermedad` | Caracter | Diagnóstico vigilado | — |
| `ano` | Integer | Año | 2000–2022 |
| `semana` | Integer | Semana epidemiológica | 1–53 |
| `diagnostic` | Caracter | CIE-10 | — |
| `tipo_dx` | Caracter | Tipo de diagnóstico | C=Confirmado, P=Probable, S=Sospechoso |
| `diresa` | Caracter | Dirección de salud notificante | — |
| `ubigeo` | Caracter | Código UBIGEO | — |
| `edad` | Caracter | Edad del paciente | — |
| `tipo_edad` | Caracter | Tipo de edad | A=Año, M=Mes, D=Días |
| `sexo` | Caracter | Sexo | M=Masculino, F=Femenino |

> 📦 **Total registros:** 604,968  
> 🔗 Fuente: [Vigilancia Epidemiológica de Malaria – datos.gob.pe](https://www.datosabiertos.gob.pe/dataset/vigilancia-epidemiol%C3%B3gica-de-malaria)

---

## 🗂️ Estructura del Repositorio

```
📦 malaria-ml-peru-2027/
│
├── 📂 data/
│   ├── raw/                          # datos_abiertos_vigilancia_malaria.csv
│   └── processed/                    # Datos limpios y transformados
│
├── 📂 notebooks/
│   ├── 01_Análisis_exploratorio_Casos_de_Malaria.ipynb
│   └── 02_MACHINE_LEARNING-PREDICCIÓN_DE_CASOS_DE_MALARIA.ipynb
│
├── 📂 dashboard/
│   ├── Cronograma_presupuesto_dashboard.pbix
│   └── Power_BI_Casos_de_Malaria.pbix
│
├── 📂 docs/
│   ├── Proyecto_de_investigación.pdf
│   ├── Proyecto_Piloto_avance_50%.pdf
│   ├── Expoandina_Final.pdf
│   └── Matriz_consistencia_operacionalización.xlsx
│
└── README.md
```

---

## 🔄 Pipeline del Proyecto

```
📥 Recolección       🔧 ETL (SCRUM)        🤖 Modelado ML         📊 Power BI          🚀 Predicción
   RENACE CSV    →  Limpieza/Transform  →  Entrenamiento/Eval  →  Visualización  →      2027
  604,968 filas      LabelEncoder           9 algoritmos           Deneb (Gantt)     Brotes por región
                     MinMaxScaler           Comparación métricas   Dashboard
```

---

## 🧪 Notebook 01 — Análisis Exploratorio

### Carga y verificación de datos
```python
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df = pd.read_csv('datos_abiertos_vigilancia_malaria.csv', ...)
print(df.shape)   # (604968, 14)
df.dtypes
df.isnull().sum()
df.nunique()
```

### Estadísticas descriptivas
- **`semana`**: min=1, max=53 — analizada con histograma, boxplot y KDE
- **`ano`**: distribución 2000–2022 via `value_counts()` + gráfico de barras horizontal
- Tasa de datos ausentes: `df.isnull().sum() / df.shape[0]`
- Atributos varianza nula identificados con función personalizada

### Visualizaciones generadas

| Visualización | Descripción |
|---|---|
| `countplot` sexo × diagnostic | Distribución de casos por género y tipo de diagnóstico |
| `countplot` enfermedad × diagnostic | Tipos de malaria por diagnóstico (CIE-10) |
| `FacetGrid` tipo_edad | Histogramas de tipo de edad por diagnóstico |
| `countplot` departamento × sexo | Casos por departamento y sexo |
| Pie chart departamento | Distribución porcentual por región (Loreto destacado) |
| Pie chart sexo | Proporción masculino (≈56%) / femenino (≈44%) |
| KDE plots | Distribución de `semana`, `diresa`, `ubigeo` por diagnóstico |
| Scatter `semana` vs. `ano` | Patrones temporales coloreados por diagnóstico |
| Pairplot | Relaciones entre variables numéricas por `diagnostic` |
| Boxplot general | Distribución de todas las variables numéricas |
| Matriz de correlación | `data.corr()` con heatmap `coolwarm` |

### Feature Engineering
```python
# Encoding de variables categóricas
lb = LabelEncoder()
df['sexo']         = lb.fit_transform(df['sexo'])
df['departamento'] = lb.fit_transform(df['departamento'])

# Escalado numérico
scaler = MinMaxScaler()
df['diresa'] = scaler.fit_transform(df[['diresa']])
df['edad']   = scaler.fit_transform(df[['edad']])

# Estratificación de semanas epidemiológicas
df["Estratos"] = [0 if i < 10 else 1 if i < 20 else
                  2 if i < 30 else 3 if i < 40 else 4
                  for i in df["semana"]]

# Binarización de edad por grupos etarios
df.loc[df['edad'] <= 15, 'edad'] = 0            # Niños
df.loc[(df['edad'] > 15) & (df['edad'] <= 30), 'edad'] = 1  # Jóvenes
# ...

# Eliminación de columnas de alta cardinalidad
df.drop(["localidad", "provincia", "distrito"], axis=1, inplace=True)
```

---

## 🏋️ Notebook 02 — Machine Learning

### Preprocesamiento común
```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Codificar variables categóricas
label_encoder = LabelEncoder()
categorical_columns = ['departamento','provincia','distrito','localidad',
                       'enfermedad','diagnostic','tipo_dx','diresa',
                       'ubigeo','tipo_edad','sexo']
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Variable objetivo: año de ocurrencia
X = data.drop(columns=['ano', 'localcod'])
y = data['ano']

# División 80% entrenamiento / 20% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Escalado estándar
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
```

### Modelos entrenados y evaluados

| # | Modelo | Librería | Parámetros clave |
|---|--------|----------|-----------------|
| 1 | **Random Forest** | `sklearn.ensemble` | `n_estimators=100`, `random_state=42` |
| 2 | **Árbol de Decisión** | `sklearn.tree` | `max_depth=3`, `random_state=42` |
| 3 | **Gradient Boosting** | `sklearn.ensemble` | `random_state=42` |
| 4 | **XGBoost** | `xgboost` | `n_estimators=100`, `random_state=42` |
| 5 | **KNN** | `sklearn.neighbors` | `n_neighbors=5` |
| 6 | **Regresión Logística** | `sklearn.linear_model` | `random_state=42` |
| 7 | **SVM** | `sklearn.svm` | `random_state=42` |
| 8 | **Red Neuronal (Keras)** | `tensorflow.keras` | `Dense(128→64)`, `Dropout(0.2)`, `epochs=50`, `Adam(lr=0.001)` |
| 9 | **MLP (sklearn)** | `sklearn.neural_network` | `random_state=42` |

### Arquitectura Red Neuronal (Keras)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(n_classes, activation='softmax')
])
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)
history = model.fit(X_train, y_train,
                    epochs=50, batch_size=32,
                    validation_data=(X_test, y_test))
```

### Métricas de evaluación
```python
from sklearn.metrics import classification_report, confusion_matrix

# Para todos los modelos:
print(classification_report(y_test, y_pred))   # Precision, Recall, F1
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')

# Feature importance (RF, DT, GB):
pd.DataFrame(model.feature_importances_,
             index=X.columns,
             columns=['importance']).sort_values('importance', ascending=False)

# Curvas de entrenamiento (Keras):
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.plot(history.history['accuracy'],     label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
```

---

## 🛠️ Instalación y Uso

### Requisitos
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow
```

### Ejecución local
```bash
# Clonar el repositorio
git clone https://github.com/<usuario>/malaria-ml-peru-2027.git
cd malaria-ml-peru-2027

# Notebook 1 — Análisis exploratorio
jupyter notebook "notebooks/01_Análisis_exploratorio_Casos_de_Malaria.ipynb"

# Notebook 2 — Modelos ML
jupyter notebook "notebooks/02_MACHINE_LEARNING-PREDICCIÓN_DE_CASOS_DE_MALARIA.ipynb"
```

> ⚠️ Los notebooks fueron desarrollados en **Google Colab**. Para ejecución local reemplaza la ruta del CSV:
> ```python
> # Reemplazar:
> df = pd.read_csv('/content/drive/MyDrive/.../datos_abiertos_vigilancia_malaria.csv')
> # Por:
> df = pd.read_csv('data/raw/datos_abiertos_vigilancia_malaria.csv')
> ```

---

## 📈 Dashboards Power BI

| Archivo | Contenido |
|---|---|
| `Power_BI_Casos_de_Malaria.pbix` | Análisis descriptivo interactivo 2000–2022 |
| `Cronograma_presupuesto_dashboard.pbix` | Diagrama de Gantt SCRUM (extensión **Deneb**) + Presupuesto |

**Hallazgos del dashboard:**
- **Loreto** concentra la mayor cantidad de casos históricos (>500,000 acumulados)
- *Malaria por P. Vivax* supera ampliamente a *P. Falciparum* en volumen
- Casos masculinos (~56%) superan a femeninos (~44%) en todos los departamentos
- Pico histórico: alrededor del año 2012–2014

---

## 💰 Presupuesto del Proyecto

| Categoría | Detalle | Total (S/.) |
|---|---|---|
| Recursos humanos | 2 investigadores | 0 |
| Bienes | Laptops (×2), materiales | 6,207 |
| Servicios | Internet, luz, agua, transporte | 1,100 |
| **TOTAL** | — | **7,307** |

*Proyecto autofinanciado.*

---

## 📅 Cronograma SCRUM

```
FASE             INICIO      FIN        DURACIÓN
─────────────────────────────────────────────────
PLANIFICACIÓN    10/04/24    17/04/24   8 días
PROCESAMIENTO    17/04/24    24/04/24   8 días
PILOTO           24/04/24    30/04/24   5 días
VISUALIZACIÓN    30/04/24    02/05/24   9 días
MODELADO         06/05/24    19/05/24   14 días
CIERRE           19/05/24    25/05/24   7 días
```

---

## 🔬 Tipo de Investigación

| Característica | Descripción |
|---|---|
| Tipo | Aplicada predictiva |
| Nivel | Explicativo |
| Diseño | No experimental, longitudinal |
| Fuente de datos | RENACE — Plataforma Nacional de Datos Abiertos |
| Período analizado | 2000–2022 |
| Registros en dataset | 604,968 |

---

## 📚 Referencias Principales

- Dukuzumuremyi (2020). *Machine learning based prediction of malaria outbreak using environment data in Rwanda.* University of Rwanda.
- Khan et al. (2024). *Predicting malaria outbreak in The Gambia using machine learning techniques.* PLOS ONE.
- Nkiruka et al. (2021). *Prediction of malaria incidence using climate variability and machine learning.* Informatics in Medicine Unlocked.
- Janiesch et al. (2021). *Machine learning and deep learning.* Electronic Markets.
- Singh et al. (2019). *Predicting Dengue Spread in San Juan and Iquitos using Machine Learning.*
- [Vigilancia Epidemiológica de Malaria – datos.gob.pe](https://www.datosabiertos.gob.pe/dataset/vigilancia-epidemiol%C3%B3gica-de-malaria)

---

## 📄 Licencia

Proyecto de uso académico — Universidad Nacional Mayor de San Marcos, 2024.

---

<div align="center">

**E.P. Estadística · UNMSM · Lima, Perú · 2024**

*"La estadística como soporte a la toma de decisiones"* — EXPOANDINA

</div>
