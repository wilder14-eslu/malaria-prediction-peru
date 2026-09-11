"""Tests de la logica de inferencia (``src/inference/predict.py``): arma el
vector de features mas reciente de un UBIGEO y predice con el campeon de
cada horizonte, sea LightGBM por cuantiles o un baseline.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.features.pipeline import build_features
from src.inference.predict import UbigeoNotFoundError, predict_for_ubigeo
from src.models.forecasting import get_feature_columns, train_quantile_models


def _synthetic_features() -> pd.DataFrame:
    # 12 semanas de casos para 2 UBIGEO, suficiente para que lags/rolling/
    # growth_rate tengan valores no nulos en la fila mas reciente.
    rows = []
    for ubigeo, dept in [("160101", "LORETO"), ("240101", "TUMBES")]:
        for week in range(1, 13):
            rows.append(
                {
                    "ubigeo": ubigeo,
                    "departamento": dept,
                    "provincia": "X",
                    "distrito": "Y",
                    "epi_year": 2024,
                    "epi_week": week,
                    "cases_falciparum": week % 3,
                    "cases_vivax": week % 2,
                    "cases_total": (week % 3) + (week % 2),
                }
            )
    canonical = pd.DataFrame(rows)
    return build_features(canonical)


def test_predict_for_ubigeo_with_baseline_champion() -> None:
    features = _synthetic_features()
    champion_config = {
        "h1": {"champion": "naive"},
        "h2": {"champion": "moving_average_4"},
    }

    result = predict_for_ubigeo("160101", features, champion_config, horizons=(1, 2))

    assert result["ubigeo"] == "160101"
    assert result["departamento"] == "LORETO"
    assert result["as_of_epi_week"] == 12
    assert result["predictions"][1]["champion"] == "naive"
    assert "point_estimate" in result["predictions"][1]
    assert result["predictions"][2]["champion"] == "moving_average_4"


def test_predict_for_ubigeo_with_lightgbm_champion(tmp_path: Path) -> None:
    features = _synthetic_features()
    feature_cols = get_feature_columns(features, horizons=(1,))
    features_with_target = features.assign(target_h1=features["cases_total"])

    # Modelo real (no un mock): entrena rapido sobre los mismos datos
    # sinteticos, para probar el camino completo de principio a fin.
    models = train_quantile_models(
        features_with_target, feature_cols, "target_h1", quantiles=(0.1, 0.5, 0.9)
    )
    model_path = tmp_path / "forecasting_h1_champion.joblib"
    joblib.dump(models, model_path)

    champion_config = {"h1": {"champion": "lightgbm_quantile"}}

    result = predict_for_ubigeo(
        "160101", features, champion_config, horizons=(1,), models_dir=tmp_path
    )

    quantiles = result["predictions"][1]["quantiles"]
    assert result["predictions"][1]["champion"] == "lightgbm_quantile"
    assert quantiles["q0.1"] <= quantiles["q0.5"] <= quantiles["q0.9"]


def test_predict_for_ubigeo_raises_for_unknown_ubigeo() -> None:
    features = _synthetic_features()
    champion_config = {"h1": {"champion": "naive"}}

    with pytest.raises(UbigeoNotFoundError):
        predict_for_ubigeo("999999", features, champion_config, horizons=(1,))
