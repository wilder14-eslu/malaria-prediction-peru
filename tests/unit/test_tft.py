"""Test estructural del Temporal Fusion Transformer (Etapa 2 del rediseno
ML). Se salta automaticamente si el extra ``dl`` no esta instalado
(``pip install -e ".[dl]"``) -- pytorch + pytorch-forecasting son pesados
(~2GB) y no son parte de las dependencias por defecto del proyecto.

No valida metricas (para eso hace falta clima real, ver
``pipelines/reference/fetch_climate_data.py``): solo confirma que la
construccion del dataset, el modelo y un paso de entrenamiento/prediccion
corren sin reventar, sobre datos sinteticos con la misma forma que produce
``src/features/climate.add_climate_features``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pytorch_forecasting", reason="requiere el extra 'dl' (pip install -e '.[dl]')")

import torch  # noqa: E402
from pytorch_forecasting import TemporalFusionTransformer  # noqa: E402

from src.models.tft import (  # noqa: E402
    CLIMATE_COLS,
    TIME_IDX_COL,
    build_dataset,
    build_validation_dataset,
    filter_known_groups,
    train_tft,
)


def _synthetic_features(n_weeks: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ubigeos = {"160101": "LORETO", "240101": "TUMBES"}
    rows = []
    for ubigeo, departamento in ubigeos.items():
        for w in range(n_weeks):
            epi_year = 2010 + w // 52
            epi_week = (w % 52) + 1
            row = {
                "ubigeo": ubigeo,
                "departamento": departamento,
                "provincia": "X",
                "distrito": "Y",
                "epi_year": epi_year,
                "epi_week": epi_week,
                "epi_index": epi_year * 52 + epi_week,
                "cases_total": max(0, int(5 + rng.normal())),
                "week_sin": np.sin(2 * np.pi * epi_week / 52),
                "week_cos": np.cos(2 * np.pi * epi_week / 52),
            }
            for col in CLIMATE_COLS:
                row[col] = rng.normal(25, 5)
            rows.append(row)
    return pd.DataFrame(rows)


def test_tft_dataset_model_and_training_run_end_to_end() -> None:
    df = _synthetic_features(n_weeks=40)
    train_cutoff = int(df[TIME_IDX_COL].quantile(0.8))
    train_df = df[df[TIME_IDX_COL] <= train_cutoff]
    full_df = df

    min_epi_index = int(train_df[TIME_IDX_COL].min())
    training_dataset = build_dataset(
        train_df, max_encoder_length=6, max_prediction_length=3, min_epi_index=min_epi_index
    )
    validation_dataset = build_validation_dataset(training_dataset, full_df, min_epi_index)

    model, _trainer = train_tft(
        training_dataset, validation_dataset, batch_size=8, max_epochs=1, patience=1
    )

    val_loader = validation_dataset.to_dataloader(train=False, batch_size=8, num_workers=0)
    predictions = model.predict(val_loader, mode="prediction")

    # 2 series (ubigeos) x max_prediction_length=3 semanas pronosticadas.
    assert predictions.shape == (2, 3)


def test_filter_known_groups_drops_ubigeos_absent_from_training() -> None:
    df = _synthetic_features(n_weeks=40)
    # "240101" nunca aparecio en entrenamiento (p.ej. su primer caso
    # registrado cae despues del corte de entrenamiento en datos reales).
    known = {"160101"}

    filtered = filter_known_groups(df, known, group_col="ubigeo")

    assert set(filtered["ubigeo"].unique()) == {"160101"}
    assert len(filtered) == len(df[df["ubigeo"] == "160101"])


def test_filter_known_groups_keeps_all_rows_when_all_groups_known() -> None:
    df = _synthetic_features(n_weeks=40)

    filtered = filter_known_groups(df, df["ubigeo"], group_col="ubigeo")

    assert len(filtered) == len(df)


def test_train_tft_resumes_from_checkpoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Entrenamiento resumible (ver docstring de train_tft): tras cortar el
    entrenamiento a 1 epoch y guardar checkpoint, retomar con
    resume_from_checkpoint no debe reventar y debe avanzar el contador de
    epoch en vez de reiniciar desde 0 -- esto es lo que hace seguro correr
    el entrenamiento real en tandas cuando el proceso se puede interrumpir
    a mitad de camino (ver run_tft_model.py).
    """
    df = _synthetic_features(n_weeks=40)
    train_cutoff = int(df[TIME_IDX_COL].quantile(0.8))
    train_df = df[df[TIME_IDX_COL] <= train_cutoff]
    full_df = df

    min_epi_index = int(train_df[TIME_IDX_COL].min())
    training_dataset = build_dataset(
        train_df, max_encoder_length=6, max_prediction_length=3, min_epi_index=min_epi_index
    )
    validation_dataset = build_validation_dataset(training_dataset, full_df, min_epi_index)

    checkpoint_dir = tmp_path / "tft_training_state"
    _model, trainer = train_tft(
        training_dataset,
        validation_dataset,
        batch_size=8,
        max_epochs=1,
        patience=5,
        checkpoint_dir=checkpoint_dir,
    )
    assert trainer.current_epoch == 1
    last_ckpt = checkpoint_dir / "last.ckpt"
    assert last_ckpt.exists()

    _model2, trainer2 = train_tft(
        training_dataset,
        validation_dataset,
        batch_size=8,
        max_epochs=2,
        patience=5,
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=str(last_ckpt),
    )
    # Retomo desde el epoch 1 (ya completado) y avanzo al 2, no reinicio en 0.
    assert trainer2.current_epoch == 2


def test_train_tft_returns_best_checkpoint_not_last_epoch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Bug real encontrado entrenando con datos reales: con early stopping,
    el modelo que queda en memoria al terminar trainer.fit() es el del
    ULTIMO epoch entrenado, no el de mejor val_loss -- si early stopping
    corta porque los ultimos epochs empeoraron, evaluar/guardar ese ultimo
    epoch da metricas peores de lo que el modelo realmente logro. train_tft
    debe devolver los pesos del mejor checkpoint (best.ckpt), no los del
    ultimo epoch en memoria.
    """
    df = _synthetic_features(n_weeks=40)
    train_cutoff = int(df[TIME_IDX_COL].quantile(0.8))
    train_df = df[df[TIME_IDX_COL] <= train_cutoff]
    full_df = df

    min_epi_index = int(train_df[TIME_IDX_COL].min())
    training_dataset = build_dataset(
        train_df, max_encoder_length=6, max_prediction_length=3, min_epi_index=min_epi_index
    )
    validation_dataset = build_validation_dataset(training_dataset, full_df, min_epi_index)

    checkpoint_dir = tmp_path / "tft_training_state"
    model, trainer = train_tft(
        training_dataset,
        validation_dataset,
        batch_size=8,
        max_epochs=3,
        patience=5,
        checkpoint_dir=checkpoint_dir,
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    assert best_model_path
    expected = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    returned_state = model.state_dict()
    expected_state = expected.state_dict()
    assert returned_state.keys() == expected_state.keys()
    for key in returned_state:
        assert torch.equal(returned_state[key], expected_state[key]), key
