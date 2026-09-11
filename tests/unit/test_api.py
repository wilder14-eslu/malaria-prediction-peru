"""Tests de la API de inferencia (``api/main.py``). Usa ``TestClient`` de
FastAPI; las funciones de carga (``build_latest_features``,
``load_champion_config``) se mockean para no depender de que exista un
pipeline de entrenamiento corrido en disco.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from src.features.pipeline import build_features


def _synthetic_features() -> pd.DataFrame:
    rows = []
    for week in range(1, 13):
        rows.append(
            {
                "ubigeo": "160101",
                "departamento": "LORETO",
                "provincia": "MAYNAS",
                "distrito": "IQUITOS",
                "epi_year": 2024,
                "epi_week": week,
                "cases_falciparum": week % 3,
                "cases_vivax": week % 2,
                "cases_total": (week % 3) + (week % 2),
            }
        )
    return build_features(pd.DataFrame(rows))


def test_health_reports_not_ready_when_startup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> None:
        raise FileNotFoundError("no canonical parquet")

    monkeypatch.setattr(api_main, "build_latest_features", _raise)
    monkeypatch.setattr(api_main, "load_champion_config", _raise)

    with TestClient(api_main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "not_ready"
    assert response.json()["champion_config_loaded"] is False


def test_predict_returns_503_when_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> None:
        raise FileNotFoundError("no canonical parquet")

    monkeypatch.setattr(api_main, "build_latest_features", _raise)
    monkeypatch.setattr(api_main, "load_champion_config", _raise)

    with TestClient(api_main.app) as client:
        response = client.get("/predict/160101")

    assert response.status_code == 503


def test_health_and_predict_when_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_main, "build_latest_features", _synthetic_features)
    monkeypatch.setattr(
        api_main,
        "load_champion_config",
        lambda: {
            "h1": {"champion": "naive"},
            "h2": {"champion": "moving_average_4"},
            "h3": {"champion": "naive"},
            "h4": {"champion": "naive"},
        },
    )

    with TestClient(api_main.app) as client:
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "ok"
        assert health_response.json()["n_ubigeos_available"] == 1

        predict_response = client.get("/predict/160101")
        assert predict_response.status_code == 200
        body = predict_response.json()
        assert body["ubigeo"] == "160101"
        assert body["departamento"] == "LORETO"
        assert body["as_of_epi_week"] == 12
        assert len(body["predictions"]) == 4
        h1 = next(p for p in body["predictions"] if p["horizon"] == 1)
        assert h1["champion"] == "naive"
        assert h1["point_estimate"] is not None
        assert h1["quantiles"] is None


def test_predict_unknown_ubigeo_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_main, "build_latest_features", _synthetic_features)
    monkeypatch.setattr(api_main, "load_champion_config", lambda: {"h1": {"champion": "naive"}})

    with TestClient(api_main.app) as client:
        response = client.get("/predict/999999")

    assert response.status_code == 404
