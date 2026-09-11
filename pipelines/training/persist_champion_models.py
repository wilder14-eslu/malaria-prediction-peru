"""Compara todos los candidatos (baselines + LightGBM por cuantiles) en el
set de validacion, por horizonte, y persiste el campeon de cada horizonte.

Metrica de seleccion: WAPE del pronostico central (mediana/q0.5 para
LightGBM). Si WAPE no esta definido (denominador 0, ej. semanas de
validacion todas en cero, como pasa con los CSV de muestra de este repo),
se usa MAE como respaldo, para que la seleccion de campeon siga
funcionando en vez de quedar indefinida.

Se persiste en dos lugares:
- Localmente: ``models/champion.json`` (que gano por horizonte, con que
  metrica, y el challenger) y ``models/forecasting_h{h}_champion.joblib``
  (solo cuando el campeon es LightGBM; un baseline es una formula, no un
  objeto que haga falta guardar).
- En MLflow: todas las corridas se loguean para trazabilidad. El modelo
  campeon LightGBM se registra en el Model Registry bajo
  ``malaria_forecasting_h{h}``.

Sobre donde vive la base de datos de MLflow (importante si el repo esta en
una carpeta sincronizada con OneDrive/Dropbox/Drive, como este proyecto):
el Model Registry necesita un backend de base de datos (SQLite aqui, no el
backend de archivos por defecto de MLflow, que no lo soporta). SQLite usa
locks de archivo que las carpetas sincronizadas en la nube suelen romper
(se confirmo en pruebas: el mismo SQLite que funciona bien fuera de una
carpeta sincronizada da "disk I/O error" dentro de una). Por eso la base
de MLflow se guarda por defecto FUERA del repo, en el home del usuario
(``~/.mlflow-malaria-prediction-peru/mlruns.db``), y no dentro de la
carpeta de OneDrive. Se puede sobreescribir con la variable de entorno
``MLFLOW_TRACKING_URI`` si se prefiere otra ubicacion o un servidor real.

Uso: ``python -m pipelines.training.persist_champion_models``
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.lightgbm
import pandas as pd
import yaml

from src.evaluation.forecasting import mae, wape
from src.features.pipeline import build_features
from src.features.spatial import add_neighbor_lag_features, load_adjacency
from src.features.targets import create_forecast_targets
from src.models.baseline import BASELINE_PREDICTORS
from src.models.forecasting import get_feature_columns, predict_quantiles, train_quantile_models
from src.preprocessing.temporal_split import walk_forward_split

DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")
ADJACENCY_PATH = Path("data/reference/district_adjacency.csv")
MODELS_DIR = Path("models")
CHAMPION_JSON_PATH = MODELS_DIR / "champion.json"

_DEFAULT_MLFLOW_DB = Path.home() / ".mlflow-malaria-prediction-peru" / "mlruns.db"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{_DEFAULT_MLFLOW_DB}")
MLFLOW_EXPERIMENT = "malaria_forecasting"


def _score(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, str]:
    """WAPE si esta definido, si no MAE como respaldo. Devuelve (valor, metrica_usada)."""
    w = wape(y_true, y_pred)
    if w == w:  # not NaN
        return w, "wape"
    return mae(y_true, y_pred), "mae"


def _load_config() -> tuple[dict[str, Any], dict[str, Any]]:
    with DATA_CONFIG_PATH.open(encoding="utf-8") as f:
        data_config = yaml.safe_load(f)
    with MODEL_CONFIG_PATH.open(encoding="utf-8") as f:
        model_config = yaml.safe_load(f)
    return data_config, model_config


def _build_dataset(
    data_config: dict[str, Any], model_config: dict[str, Any], horizons: tuple[int, ...]
) -> pd.DataFrame:
    canonical = pd.read_parquet(data_config["gold"]["canonical_weekly_cases"])
    features = build_features(canonical)
    if ADJACENCY_PATH.exists():
        adjacency = load_adjacency(str(ADJACENCY_PATH))
        features = add_neighbor_lag_features(features, adjacency)
    target_col = model_config["forecasting"]["target"]
    return create_forecast_targets(features, target_col=target_col, horizons=horizons)


def main() -> None:
    data_config, model_config = _load_config()
    horizons = tuple(model_config["forecasting"]["horizons"])
    quantiles = tuple(model_config["forecasting"]["quantiles"])
    validation = model_config["validation"]

    features = _build_dataset(data_config, model_config, horizons)
    feature_cols = get_feature_columns(features, horizons=horizons)
    train, val, _ = walk_forward_split(
        features,
        train_end_year=validation["train_end_year"],
        val_end_year=validation["val_end_year"],
        test_end_year=validation["test_end_year"],
    )

    if MLFLOW_TRACKING_URI.startswith("sqlite:///"):
        Path(MLFLOW_TRACKING_URI.removeprefix("sqlite:///")).parent.mkdir(
            parents=True, exist_ok=True
        )
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    MODELS_DIR.mkdir(exist_ok=True)
    champion_info: dict[str, Any] = {}

    for h in horizons:
        target = f"target_h{h}"
        y_val_true = val[target]
        candidates: dict[str, dict[str, Any]] = {}

        for name, predict_fn in BASELINE_PREDICTORS.items():
            preds = predict_fn(val, horizons=(h,))
            score, metric_used = _score(y_val_true, preds[h])
            with mlflow.start_run(run_name=f"h{h}_{name}"):
                mlflow.log_params({"horizon": h, "candidate": name, "kind": "baseline"})
                mlflow.log_metric(metric_used, score)
            candidates[name] = {"score": score, "metric": metric_used, "kind": "baseline"}

        quantile_models = train_quantile_models(train, feature_cols, target, quantiles)
        val_preds = predict_quantiles(quantile_models, val, feature_cols)
        lgbm_score, lgbm_metric = _score(y_val_true, val_preds["q0.5"])
        with mlflow.start_run(run_name=f"h{h}_lightgbm_quantile") as run:
            mlflow.log_params({"horizon": h, "candidate": "lightgbm_quantile", "kind": "ml"})
            mlflow.log_metric(lgbm_metric, lgbm_score)
            # Flavor nativo de LightGBM (no mlflow.sklearn): serializa con el
            # formato propio de LightGBM en vez de pickle/skops, que por
            # defecto rechaza deserializar tipos de terceros (Booster,
            # LGBMRegressor) como "no confiables".
            mlflow.lightgbm.log_model(quantile_models[0.5], name="model_q0_5")
            run_id = run.info.run_id
        candidates["lightgbm_quantile"] = {
            "score": lgbm_score,
            "metric": lgbm_metric,
            "kind": "ml",
            "run_id": run_id,
            "models": quantile_models,
        }

        ranked = sorted(candidates.items(), key=lambda kv: kv[1]["score"])
        champion_name, champion_data = ranked[0]
        challenger_name = ranked[1][0] if len(ranked) > 1 else None

        entry: dict[str, Any] = {
            "champion": champion_name,
            "champion_score": champion_data["score"],
            "metric": champion_data["metric"],
            "challenger": challenger_name,
            "all_candidates": {k: v["score"] for k, v in candidates.items()},
        }

        if champion_data["kind"] == "ml":
            model_path = MODELS_DIR / f"forecasting_h{h}_champion.joblib"
            joblib.dump(champion_data["models"], model_path)
            entry["local_path"] = str(model_path)
            model_uri = f"runs:/{champion_data['run_id']}/model_q0_5"
            try:
                mlflow.register_model(model_uri, f"malaria_forecasting_h{h}")
            except Exception as exc:  # noqa: BLE001 -- registro es best-effort, no debe tumbar el pipeline
                print(f"Aviso: no se pudo registrar en MLflow Model Registry: {exc}")

        champion_info[f"h{h}"] = entry
        print(
            f"Horizonte {h}: campeon = {champion_name} "
            f"({champion_data['metric']}={champion_data['score']:.4f}), "
            f"challenger = {challenger_name}"
        )

    with CHAMPION_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(champion_info, f, indent=2, ensure_ascii=False)
    print(f"\nchampion.json escrito en {CHAMPION_JSON_PATH}")


if __name__ == "__main__":
    main()
