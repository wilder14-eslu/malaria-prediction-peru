"""Temporal Fusion Transformer (Etapa 2 del rediseno ML, ver
``docs/architecture.md``): un solo modelo que aprende su propia
representacion temporal por UBIGEO (en vez de un modelo separado por
horizonte como en Etapa 1/LightGBM) y que puede usar clima como covariable.

Diferencia deliberada con Etapa 1: aca NO se le pasan los ``lag_*`` /
``rolling_*`` ya calculados en ``src/features/temporal.py`` -- el
encoder-decoder del TFT aprende su propia representacion de la historia
reciente de la serie a partir del target crudo. Pasarle ademas los lags ya
calculados seria redundante (el mismo tipo de informacion dos veces) y es
una practica que la literatura de TFT no recomienda; el modelo si recibe
clima (no observable de antemano, ver nota de "unknown reals" abajo) y
estacionalidad calendario (si conocida de antemano).

Requiere el extra ``dl`` (``pip install -e ".[dl]"``): pytorch +
pytorch-forecasting + lightning, pesado (~2GB), por eso separado del resto
de dependencias.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

# Mismas variables climaticas que produce src/features/climate.py.
CLIMATE_COLS = ["t2m_mean", "t2m_max", "t2m_min", "precip_mm_sum", "rh2m_mean"]
KNOWN_CALENDAR_COLS = ["week_sin", "week_cos"]
TARGET_COL = "cases_total"
GROUP_COL = "ubigeo"
TIME_IDX_COL = "epi_index"
STATIC_CATEGORICALS = ["departamento"]

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


def prepare_dataframe(
    df: pd.DataFrame, min_epi_index: int | None = None
) -> tuple[pd.DataFrame, int]:
    """Deja la tabla lista para ``TimeSeriesDataSet``: agrega ``time_idx``
    (``epi_index`` desplazado a partir de ``min_epi_index``, 0-based como
    pytorch-forecasting espera) y fuerza los dtypes que la libreria exige.

    Se expone aparte de ``build_dataset`` porque el dataset de
    validacion/prediccion NO se construye con ``TimeSeriesDataSet``
    directamente sino con ``TimeSeriesDataSet.from_dataset(training, df,
    ...)`` (para reusar los encoders/normalizadores ajustados en train) --
    pero ese ``df`` necesita pasar por la misma preparacion, si no
    revienta con el mismo tipo de error de dtype que ``build_dataset``
    evita aca.
    """
    df = df.copy()
    if min_epi_index is None:
        min_epi_index = int(df[TIME_IDX_COL].min())
    df["time_idx"] = df[TIME_IDX_COL] - min_epi_index
    df[GROUP_COL] = df[GROUP_COL].astype(str)
    # cases_total llega como int (conteos); el GroupNormalizer con
    # transformation="softplus" necesita un dtype de punto flotante.
    df[TARGET_COL] = df[TARGET_COL].astype(float)
    return df, min_epi_index


def filter_known_groups(
    df: pd.DataFrame, known_groups: pd.Series | set[str] | list[str], group_col: str = GROUP_COL
) -> pd.DataFrame:
    """Filtra ``df`` a solo las filas cuyo ``group_col`` (UBIGEO) aparece en
    ``known_groups`` (tipicamente, los UBIGEO presentes en el set de
    entrenamiento).

    Necesario porque ``TimeSeriesDataSet`` ajusta el encoder categorico de
    ``group_ids`` SOLO con los datos de entrenamiento (asi evita fuga de
    informacion de validacion/test hacia el encoder). Un UBIGEO cuyo primer
    caso registrado cae despues del corte de entrenamiento pero antes del
    corte de validacion -- en los datos reales de este proyecto, un numero
    chico de distritos (2 de 1639) -- no tiene ninguna fila en ``train_df``,
    y al aparecer en el dataframe pasado a ``build_validation_dataset``
    revienta con ``KeyError: Unknown category '<ubigeo>' encountered``.

    Esto no es un bug a parchear silenciosamente sino una limitacion real
    del enfoque: un modelo con un embedding categorico por serie (como el
    TFT aca) estructuralmente no puede personalizar una prediccion para una
    entidad que nunca vio en entrenamiento, sin importar cuanta historia
    tenga esa entidad en si misma. La alternativa (``add_nan=True`` en el
    encoder de ``group_ids``, que le asignaria un embedding generico
    "desconocido") evitaria el crash pero mezclaria silenciosamente esos
    distritos con una representacion que no es la suya -- se prefiere
    excluirlos de forma explicita y que el caller reporte cuantos son.
    """
    known = {str(g) for g in known_groups}
    return df[df[group_col].astype(str).isin(known)].copy()


def build_dataset(
    df: pd.DataFrame,
    max_encoder_length: int,
    max_prediction_length: int,
    min_epi_index: int | None = None,
    predict_mode: bool = False,
) -> TimeSeriesDataSet:
    """Construye el ``TimeSeriesDataSet`` de entrenamiento a partir de la
    tabla de features (post ``complete_weekly_grid`` + clima unido)."""
    df, _ = prepare_dataframe(df, min_epi_index)

    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=TARGET_COL,
        group_ids=[GROUP_COL],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=STATIC_CATEGORICALS,
        time_varying_known_reals=["time_idx", *KNOWN_CALENDAR_COLS],
        # Clima: no se conoce de antemano (no hay pronostico climatico en
        # este pipeline), solo se observa historicamente -> unknown real.
        time_varying_unknown_reals=[TARGET_COL, *CLIMATE_COLS],
        target_normalizer=GroupNormalizer(groups=[GROUP_COL], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
        predict_mode=predict_mode,
    )


def build_validation_dataset(
    training_dataset: TimeSeriesDataSet, df: pd.DataFrame, min_epi_index: int
) -> TimeSeriesDataSet:
    """Construye el dataset de validacion/prediccion reusando los
    encoders/normalizadores ya ajustados en ``training_dataset`` (patron
    estandar de pytorch-forecasting), con la misma preparacion de dtypes
    que ``build_dataset``.
    """
    prepared, _ = prepare_dataframe(df, min_epi_index)
    return TimeSeriesDataSet.from_dataset(
        training_dataset, prepared, predict=True, stop_randomization=True
    )


def build_model(
    training_dataset: TimeSeriesDataSet,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    learning_rate: float = 0.03,
    hidden_size: int = 16,
    attention_head_size: int = 1,
    dropout: float = 0.1,
) -> TemporalFusionTransformer:
    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        loss=QuantileLoss(list(quantiles)),
        log_interval=0,
    )


def train_tft(
    train_dataset: TimeSeriesDataSet,
    val_dataset: TimeSeriesDataSet,
    batch_size: int = 64,
    max_epochs: int = 30,
    patience: int = 5,
    accelerator: str = "cpu",
    trainer_overrides: dict[str, Any] | None = None,
    checkpoint_dir: Path | None = None,
    resume_from_checkpoint: str | None = None,
) -> tuple[TemporalFusionTransformer, pl.Trainer]:
    """Entrena el TFT con early stopping sobre la perdida de validacion.

    ``checkpoint_dir`` / ``resume_from_checkpoint`` habilitan entrenamiento
    resumible por epoch: en datos reales a esta escala, un solo epoch puede
    tardar minutos u horas segun el hardware (ver nota de timing real en
    ``run_tft_model.py``), y un entorno de ejecucion en la nube puede
    suspenderse/perder el proceso en medio de una corrida larga. Con
    ``checkpoint_dir`` seteado, Lightning guarda el ultimo epoch completo
    (``last.ckpt``) ademas del mejor por val_loss (``best.ckpt``) despues de
    cada epoch; pasando ese ``last.ckpt`` de vuelta como
    ``resume_from_checkpoint`` en la siguiente invocacion, el entrenamiento
    continua exactamente desde ahi (pesos, optimizador y contador de epoch),
    sin perder el progreso ya hecho.

    Sin ``checkpoint_dir`` (por defecto), no se activa checkpointing de
    Lightning -- el caller es responsable de persistir el modelo final con
    ``torch.save(model.state_dict(), ...)``, y no quedan directorios
    ``./checkpoints/`` sueltos (ver ``enable_checkpointing`` abajo).

    Importante sobre que modelo se devuelve: con ``checkpoint_dir`` seteado
    (y por lo tanto ``ModelCheckpoint`` activo), al terminar ``trainer.fit``
    esta funcion recarga los pesos del MEJOR checkpoint por ``val_loss``
    (``best.ckpt``) antes de devolver el modelo -- no los del ultimo epoch
    entrenado. Esto importa en concreto con early stopping: el early
    stopping para el entrenamiento recien despues de ``patience`` epochs SIN
    mejora, asi que el modelo que queda en memoria al final es el del
    ULTIMO epoch (ya degradado/con peor val_loss que el mejor visto), no el
    mejor. Evaluar ese ultimo epoch en vez del mejor da metricas
    artificialmente peores y no representa la capacidad real del modelo
    (se detecto exactamente este caso entrenando con datos reales: el
    ultimo epoch daba WAPE=3.34, el mejor epoch WAPE=1.33).
    """
    train_loader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    model = build_model(train_dataset)

    callbacks: list[Any] = [EarlyStopping(monitor="val_loss", patience=patience, mode="min")]
    enable_checkpointing = False
    if checkpoint_dir is not None:
        enable_checkpointing = True
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="best",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=True,
            )
        )

    trainer_kwargs: dict[str, Any] = {
        "max_epochs": max_epochs,
        "accelerator": accelerator,
        "gradient_clip_val": 0.1,
        "callbacks": callbacks,
        "enable_progress_bar": False,
        "logger": False,
        # El estado del modelo se guarda explicitamente con torch.save en
        # el pipeline (ver run_tft_model.py); sin checkpoint_dir, Lightning
        # ademas escribiria sus propios checkpoints en ./checkpoints/.
        "enable_checkpointing": enable_checkpointing,
    }
    if trainer_overrides:
        trainer_kwargs.update(trainer_overrides)

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_from_checkpoint,
        # PyTorch >=2.6 cambio el default de torch.load a weights_only=True,
        # que rompe la carga de checkpoints de Lightning (incluyen objetos
        # no-tensor: normalizadores de pytorch-forecasting, estado del
        # optimizador, etc.). weights_only=False es seguro aca porque el
        # checkpoint es siempre el propio (escrito por este mismo pipeline
        # en checkpoint_dir), nunca uno de origen externo/no confiable.
        weights_only=False if resume_from_checkpoint else None,
    )

    checkpoint_callback = trainer.checkpoint_callback
    if isinstance(checkpoint_callback, ModelCheckpoint) and checkpoint_callback.best_model_path:
        model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_callback.best_model_path)

    return model, trainer
