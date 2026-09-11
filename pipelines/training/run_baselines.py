"""Entrena (sin parametros reales, son heuristicas) y evalua los baselines
multi-horizonte sobre el split de validacion. Escribe
``data/gold/predictions/baseline_metrics.csv``.

Uso: ``python -m pipelines.training.run_baselines``
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.forecasting import mae, rmse, wape
from src.features.pipeline import build_features
from src.features.targets import create_forecast_targets
from src.models.baseline import BASELINE_PREDICTORS
from src.preprocessing.temporal_split import walk_forward_split

DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")
OUTPUT_PATH = Path("data/gold/predictions/baseline_metrics.csv")


def main() -> None:
    with DATA_CONFIG_PATH.open(encoding="utf-8") as f:
        data_config = yaml.safe_load(f)
    with MODEL_CONFIG_PATH.open(encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    horizons = tuple(model_config["forecasting"]["horizons"])
    target_col = model_config["forecasting"]["target"]
    validation = model_config["validation"]

    canonical = pd.read_parquet(data_config["gold"]["canonical_weekly_cases"])
    features = build_features(canonical)
    features = create_forecast_targets(features, target_col=target_col, horizons=horizons)

    _, val, _ = walk_forward_split(
        features,
        train_end_year=validation["train_end_year"],
        val_end_year=validation["val_end_year"],
        test_end_year=validation["test_end_year"],
    )

    rows = []
    for model_name, predict_fn in BASELINE_PREDICTORS.items():
        predictions = predict_fn(val, horizons=horizons)
        for h in horizons:
            y_true = val[f"target_h{h}"]
            y_pred = predictions[h]
            rows.append(
                {
                    "model": model_name,
                    "horizon": h,
                    "mae": mae(y_true, y_pred),
                    "rmse": rmse(y_true, y_pred),
                    "wape": wape(y_true, y_pred),
                    "n_obs": int(pd.concat([y_true, y_pred], axis=1).dropna().shape[0]),
                }
            )

    metrics = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_PATH, index=False)
    print(metrics.to_string(index=False))
    print(f"\nMetricas escritas en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
