"""Entrena LightGBM con salida por cuantiles (Etapa 1 del rediseno ML) por
horizonte, y evalua contra el split de validacion con pinball loss y
cobertura del intervalo. Escribe ``data/gold/predictions/quantile_model_metrics.csv``.

Si existe ``data/reference/district_adjacency.csv`` (generado por
``pipelines/reference/build_district_adjacency.py``), tambien agrega las
features espaciales de vecinos. Si no existe, sigue sin ellas (no es un
requisito duro para esta etapa basica).

Uso: ``python -m pipelines.training.run_ml_models``
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.forecasting import interval_coverage, pinball_loss
from src.features.pipeline import build_features
from src.features.spatial import add_neighbor_lag_features, load_adjacency
from src.features.targets import create_forecast_targets
from src.models.forecasting import get_feature_columns, predict_quantiles, train_quantile_models
from src.preprocessing.temporal_split import walk_forward_split

DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")
ADJACENCY_PATH = Path("data/reference/district_adjacency.csv")
OUTPUT_PATH = Path("data/gold/predictions/quantile_model_metrics.csv")


def main() -> None:
    with DATA_CONFIG_PATH.open(encoding="utf-8") as f:
        data_config = yaml.safe_load(f)
    with MODEL_CONFIG_PATH.open(encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    horizons = tuple(model_config["forecasting"]["horizons"])
    quantiles = tuple(model_config["forecasting"]["quantiles"])
    target_col = model_config["forecasting"]["target"]
    validation = model_config["validation"]

    canonical = pd.read_parquet(data_config["gold"]["canonical_weekly_cases"])
    features = build_features(canonical)

    if ADJACENCY_PATH.exists():
        adjacency = load_adjacency(str(ADJACENCY_PATH))
        features = add_neighbor_lag_features(features, adjacency)
    else:
        print(f"Aviso: {ADJACENCY_PATH} no existe, se entrena sin features espaciales.")

    features = create_forecast_targets(features, target_col=target_col, horizons=horizons)
    feature_cols = get_feature_columns(features, horizons=horizons)

    train, val, _ = walk_forward_split(
        features,
        train_end_year=validation["train_end_year"],
        val_end_year=validation["val_end_year"],
        test_end_year=validation["test_end_year"],
    )

    low_q, high_q = min(quantiles), max(quantiles)
    rows = []
    for h in horizons:
        target = f"target_h{h}"
        models = train_quantile_models(train, feature_cols, target, quantiles)
        val_preds = predict_quantiles(models, val, feature_cols)

        y_true = val[target]
        for q in quantiles:
            rows.append(
                {
                    "model": "lightgbm_quantile",
                    "horizon": h,
                    "quantile": q,
                    "pinball_loss": pinball_loss(y_true, val_preds[f"q{q}"], quantile=q),
                }
            )
        rows.append(
            {
                "model": "lightgbm_quantile",
                "horizon": h,
                "quantile": f"coverage_{low_q}_{high_q}",
                "pinball_loss": interval_coverage(
                    y_true, val_preds[f"q{low_q}"], val_preds[f"q{high_q}"]
                ),
            }
        )

    metrics = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_PATH, index=False)
    print(metrics.to_string(index=False))
    print(f"\nMetricas escritas en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
