"""Test de la logica de seleccion de metrica para Champion/Challenger:
WAPE cuando esta definido, MAE como respaldo cuando no (ej. semanas de
validacion todas en cero, donde WAPE es 0/0).
"""

from __future__ import annotations

import pandas as pd

from pipelines.training.persist_champion_models import _score


def test_score_uses_wape_when_defined() -> None:
    y_true = pd.Series([10, 20])
    y_pred = pd.Series([12, 18])

    value, metric = _score(y_true, y_pred)

    assert metric == "wape"
    assert value > 0


def test_score_falls_back_to_mae_when_wape_undefined() -> None:
    # Todo el y_true es 0: WAPE es 0/0 (NaN), debe caer a MAE.
    y_true = pd.Series([0, 0, 0])
    y_pred = pd.Series([1, 0, 2])

    value, metric = _score(y_true, y_pred)

    assert metric == "mae"
    assert value == 1.0  # (1+0+2)/3
