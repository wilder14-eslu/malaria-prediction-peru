"""Logica de inferencia para la API (``api/main.py``): dado un UBIGEO,
arma su vector de features mas reciente y predice con el modelo campeon de
cada horizonte (segun ``models/champion.json``, escrito por
``pipelines/training/persist_champion_models.py``).

Separado de ``api/`` a proposito: esto es logica de negocio testeable sin
levantar FastAPI, siguiendo la misma separacion libreria (``src/``) vs.
orquestacion/presentacion (``pipelines/``, ``api/``) del resto del proyecto.

Optimizacion de memoria (v2): en vez de recalcular toda la tabla de features
desde la tabla canonica (lo cual consumia ~500MB de RAM por el
``complete_weekly_grid`` + spatial joins sobre 2.1M filas), ahora carga un
parquet pre-materializado (``data/gold/features/materialized_features.parquet``,
generado por ``pipelines/training/run_ml_models.py``). Esto reduce el uso
de RAM a ~40MB, permitiendo correr en el free tier de Render (512MB).

Fallback: si el parquet materializado no existe, se recalcula como antes
(para desarrollo local con mas RAM disponible).
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from src.models.baseline import BASELINE_PREDICTORS
from src.models.forecasting import get_feature_columns, predict_quantiles

DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")
ADJACENCY_PATH = Path("data/reference/district_adjacency.csv")
MODELS_DIR = Path("models")
CHAMPION_JSON_PATH = MODELS_DIR / "champion.json"


class UbigeoNotFoundError(KeyError):
    """El UBIGEO pedido no aparece en la tabla de features (no hay historia
    de casos registrada para el en la fuente de vigilancia)."""


def _downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce el uso de memoria convirtiendo float64 → float32 e int64 → int32
    donde sea seguro. En la tabla de features de malaria, los valores son
    counts y ratios pequenos, asi que float32 e int32 sobran.
    """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def build_latest_features(
    data_config_path: Path = DATA_CONFIG_PATH, adjacency_path: Path = ADJACENCY_PATH
) -> pd.DataFrame:
    """Carga la tabla de features lista para inferencia.

    Estrategia (v2, optimizada para 512MB de RAM):
    1. Si existe el parquet materializado (generado por run_ml_models),
       lo carga directamente — rapido (~2-3s) y liviano (~40MB de RAM).
    2. Si no existe (desarrollo local, primera vez), recalcula desde la
       tabla canonica como antes (fallback, ~25-30s, ~500MB de RAM).
    """
    with data_config_path.open(encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    materialized_path = data_config.get("gold", {}).get("materialized_features")
    if materialized_path and Path(materialized_path).exists():
        features = pd.read_parquet(materialized_path)
        features = _downcast_dataframe(features)
        gc.collect()
        return features

    # Fallback: recalcular (necesita mas RAM, OK para desarrollo local)
    from src.features.pipeline import build_features
    from src.features.spatial import add_neighbor_lag_features, load_adjacency

    canonical = pd.read_parquet(data_config["gold"]["canonical_weekly_cases"])
    features = build_features(canonical)
    del canonical
    gc.collect()

    if adjacency_path.exists():
        adjacency = load_adjacency(str(adjacency_path))
        features = add_neighbor_lag_features(features, adjacency)
        del adjacency
        gc.collect()

    features = _downcast_dataframe(features)
    gc.collect()
    return features


def load_champion_config(path: Path = CHAMPION_JSON_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro {path}. Corre primero "
            "'python -m pipelines.training.persist_champion_models' para "
            "entrenar y seleccionar los modelos campeon."
        )
    with path.open(encoding="utf-8") as f:
        return dict(json.load(f))


def predict_for_ubigeo(
    ubigeo: str,
    features_df: pd.DataFrame,
    champion_config: dict[str, Any],
    horizons: tuple[int, ...] = (1, 2, 3, 4),
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    """Predice, para un UBIGEO, los casos esperados en cada horizonte con el
    modelo campeon correspondiente (LightGBM por cuantiles, o un baseline si
    ese fue el campeon para ese horizonte en el ultimo entrenamiento).

    Devuelve un dict listo para serializar en la respuesta de la API:
    ``{"ubigeo", "departamento", "provincia", "distrito", "as_of_epi_year",
    "as_of_epi_week", "predictions": {horizonte: {...}}}``.

    Lanza ``UbigeoNotFoundError`` si el UBIGEO no aparece en ``features_df``
    (no hay historia de casos registrada para el).
    """
    ubigeo_rows = features_df.loc[features_df["ubigeo"] == ubigeo].sort_values("epi_index")
    if ubigeo_rows.empty:
        raise UbigeoNotFoundError(
            f"UBIGEO '{ubigeo}' no tiene historia en la fuente de vigilancia."
        )
    latest = ubigeo_rows.iloc[-1]
    feature_cols = get_feature_columns(features_df, horizons=horizons)

    predictions: dict[int, dict[str, Any]] = {}
    for h in horizons:
        entry = champion_config[f"h{h}"]
        champion_name = entry["champion"]

        if champion_name == "lightgbm_quantile":
            model_path = models_dir / f"forecasting_h{h}_champion.joblib"
            quantile_models = joblib.load(model_path)
            # .loc con el indice original (no latest[feature_cols].to_frame().T)
            # para que las columnas conserven su dtype numerico real: una fila
            # sacada de una Serie mixta (numeros + strings) queda "object" y
            # LightGBM la puede rechazar o tratarla distinto que en entrenamiento.
            latest_row_df = ubigeo_rows.loc[[latest.name], feature_cols]
            quantile_preds = predict_quantiles(quantile_models, latest_row_df, feature_cols)
            predictions[h] = {
                "champion": champion_name,
                "quantiles": {
                    col: float(val) for col, val in quantile_preds.iloc[0].to_dict().items()
                },
            }
        else:
            baseline_fn = BASELINE_PREDICTORS[champion_name]
            baseline_preds = baseline_fn(ubigeo_rows, horizons=(h,))
            point_estimate = float(baseline_preds[h].iloc[-1])
            predictions[h] = {"champion": champion_name, "point_estimate": point_estimate}

    return {
        "ubigeo": ubigeo,
        "departamento": str(latest["departamento"]),
        "provincia": str(latest["provincia"]),
        "distrito": str(latest["distrito"]),
        "as_of_epi_year": int(latest["epi_year"]),
        "as_of_epi_week": int(latest["epi_week"]),
        "predictions": predictions,
    }
