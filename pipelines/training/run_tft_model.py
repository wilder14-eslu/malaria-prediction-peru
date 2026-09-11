"""Entrena el Temporal Fusion Transformer (Etapa 2 del rediseno ML, ver
docs/architecture.md) sobre la tabla canonica + clima, y evalua contra el
campeon de Etapa 1 (LightGBM, ver ``persist_champion_models.py``) en el set
de validacion.

Requiere:
- El extra ``dl`` instalado (``pip install -e ".[dl]"``).
- Clima descargado: ``python pipelines/reference/fetch_climate_data.py``
  (ver ese script -- necesita salida a internet, se corre aparte del resto
  del pipeline).

Uso: ``python -m pipelines.training.run_tft_model``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from src.evaluation.forecasting import mae, wape
from src.features.climate import add_climate_features, aggregate_climate_to_weekly
from src.features.pipeline import build_features
from src.features.spatial import add_neighbor_lag_features, load_adjacency
from src.models.tft import (
    GROUP_COL,
    TIME_IDX_COL,
    build_dataset,
    build_validation_dataset,
    filter_known_groups,
    train_tft,
)

DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")
ADJACENCY_PATH = Path("data/reference/district_adjacency.csv")
CLIMATE_DAILY_PATH = Path("data/raw/climate/climate_daily_by_department.csv")
MODELS_DIR = Path("models")
TFT_CHECKPOINT_PATH = MODELS_DIR / "tft_champion.ckpt"
# Estado de entrenamiento resumible por epoch (ver docstring de train_tft en
# src/models/tft.py) -- no es el artefacto final, por eso vive aparte y esta
# en .gitignore.
TRAINING_STATE_DIR = MODELS_DIR / "tft_training_state"
LAST_CHECKPOINT_PATH = TRAINING_STATE_DIR / "last.ckpt"


def _load_config() -> tuple[dict[str, Any], dict[str, Any]]:
    with DATA_CONFIG_PATH.open(encoding="utf-8") as f:
        data_config = yaml.safe_load(f)
    with MODEL_CONFIG_PATH.open(encoding="utf-8") as f:
        model_config = yaml.safe_load(f)
    return data_config, model_config


def _build_dataset_with_climate(data_config: dict[str, Any]) -> pd.DataFrame:
    if not CLIMATE_DAILY_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro {CLIMATE_DAILY_PATH}. Corre primero "
            "'python pipelines/reference/fetch_climate_data.py' (necesita "
            "salida a internet; ver ese script para el detalle)."
        )

    canonical = pd.read_parquet(data_config["gold"]["canonical_weekly_cases"])
    features = build_features(canonical)
    if ADJACENCY_PATH.exists():
        adjacency = load_adjacency(str(ADJACENCY_PATH))
        features = add_neighbor_lag_features(features, adjacency)

    climate_daily = pd.read_csv(CLIMATE_DAILY_PATH)
    climate_weekly = aggregate_climate_to_weekly(climate_daily)
    return add_climate_features(features, climate_weekly)


def main() -> None:
    data_config, model_config = _load_config()
    validation = model_config["validation"]
    tft_config = model_config["tft"]

    features = _build_dataset_with_climate(data_config)

    train_cutoff = validation["train_end_year"] * 52 + 52
    val_cutoff = validation["val_end_year"] * 52 + 52
    train_df = features[features[TIME_IDX_COL] <= train_cutoff].copy()
    full_df = features[features[TIME_IDX_COL] <= val_cutoff].copy()

    min_epi_index = int(train_df[TIME_IDX_COL].min())
    training_dataset = build_dataset(
        train_df,
        max_encoder_length=tft_config["max_encoder_length"],
        max_prediction_length=tft_config["max_prediction_length"],
        min_epi_index=min_epi_index,
    )

    # Ver docstring de filter_known_groups(): un UBIGEO sin ninguna fila en
    # train_df (su primer caso registrado cae entre el corte de
    # entrenamiento y el de validacion) hace reventar el encoder categorico
    # de group_ids si no se excluye antes de construir el dataset de
    # validacion.
    ubigeos_val_antes = full_df[GROUP_COL].astype(str).nunique()
    full_df = filter_known_groups(full_df, train_df[GROUP_COL], group_col=GROUP_COL)
    n_excluidos = ubigeos_val_antes - full_df[GROUP_COL].astype(str).nunique()
    if n_excluidos:
        print(
            f"Aviso: {n_excluidos} UBIGEO en el rango de validacion no tienen "
            "ninguna fila en el set de entrenamiento (su primer caso "
            "registrado es posterior al corte de entrenamiento) -- se "
            "excluyen del set de validacion. Ver docstring de "
            "filter_known_groups() en src/models/tft.py."
        )

    validation_dataset = build_validation_dataset(training_dataset, full_df, min_epi_index)

    MODELS_DIR.mkdir(exist_ok=True)
    TRAINING_STATE_DIR.mkdir(exist_ok=True)
    resume_from = str(LAST_CHECKPOINT_PATH) if LAST_CHECKPOINT_PATH.exists() else None
    if resume_from:
        print(f"Retomando entrenamiento desde checkpoint existente: {resume_from}")
    model, _trainer = train_tft(
        training_dataset,
        validation_dataset,
        batch_size=tft_config["batch_size"],
        max_epochs=tft_config["max_epochs"],
        patience=tft_config["patience"],
        checkpoint_dir=TRAINING_STATE_DIR,
        resume_from_checkpoint=resume_from,
    )
    torch.save(model.state_dict(), TFT_CHECKPOINT_PATH)

    val_loader = validation_dataset.to_dataloader(
        train=False, batch_size=tft_config["batch_size"], num_workers=0
    )
    predictions = model.predict(val_loader, mode="prediction")
    actuals = torch.cat([y[0] for _x, y in iter(val_loader)])

    y_true = pd.Series(actuals.numpy().ravel())
    y_pred = pd.Series(predictions.numpy().ravel())
    print(
        f"TFT (Etapa 2) en validacion: WAPE={wape(y_true, y_pred):.4f} "
        f"MAE={mae(y_true, y_pred):.4f}"
    )
    print(f"Checkpoint guardado en {TFT_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
