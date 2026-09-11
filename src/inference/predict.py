"""Logica de inferencia para la API (``api/main.py``): dado un UBIGEO,
arma su vector de features mas reciente y predice con el modelo campeon de
cada horizonte (segun ``models/champion.json``, escrito por
``pipelines/training/persist_champion_models.py``).

Separado de ``api/`` a proposito: esto es logica de negocio testeable sin
levantar FastAPI, siguiendo la misma separacion libreria (``src/``) vs.
orquestacion/presentacion (``pipelines/``, ``api/``) del resto del proyecto.

Limitacion conocida de esta primera version: las features (lags, rolling,
vecinos espaciales) se recalculan en memoria a partir de la tabla canonica
completa cada vez que se llama ``build_latest_features`` -- no hay un feature
store real con actualizacion incremental. Medido con los datos reales
completos (143502 registros, 1645 UBIGEO): ~25-30 segundos. Se paga UNA
sola vez al arrancar la API (``api/main.py`` lo hace en el lifespan, no en
cada request), asi que no afecta la latencia de ``/predict``, pero si el
tiempo de arranque -- a revisar (cachear a un parquet materializado, o
vectorizar el loop por UBIGEO en ``complete_weekly_grid``) si el dataset
crece mucho o el arranque en frio se vuelve un problema (ver Fase 7 en el
README).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from src.features.pipeline import build_features
from src.features.spatial import add_neighbor_lag_features, load_adjacency
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


def build_latest_features(
    data_config_path: Path = DATA_CONFIG_PATH, adjacency_path: Path = ADJACENCY_PATH
) -> pd.DataFrame:
    """Reconstruye la tabla de features completa (misma logica que usan los
    pipelines de entrenamiento) a partir de la tabla canonica. La fila mas
    reciente de cada UBIGEO es el vector de entrada para prediccion.
    """
    with data_config_path.open(encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    canonical = pd.read_parquet(data_config["gold"]["canonical_weekly_cases"])
    features = build_features(canonical)
    if adjacency_path.exists():
        adjacency = load_adjacency(str(adjacency_path))
        features = add_neighbor_lag_features(features, adjacency)
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
