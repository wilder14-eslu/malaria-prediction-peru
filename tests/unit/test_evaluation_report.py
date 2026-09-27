"""Tests de las metricas estadisticas del reporte de evaluacion."""

import numpy as np

from pipelines.evaluation.run_evaluation_report import diebold_mariano, pinball, wape, wis


def test_wape_perfect_and_known_value():
    y = np.array([10.0, 0.0, 5.0])
    assert wape(y, y) == 0.0
    assert wape(y, np.array([8.0, 1.0, 5.0])) == 3.0 / 15.0


def test_pinball_is_asymmetric():
    y = np.array([10.0])
    # Subestimar el q0.9 cuesta 9 veces mas que sobreestimarlo en la misma magnitud
    under = pinball(y, np.array([9.0]), 0.9)
    over = pinball(y, np.array([11.0]), 0.9)
    assert np.isclose(under / over, 9.0)


def test_wis_reduces_to_scaled_absolute_error_for_degenerate_interval():
    y = np.array([4.0, 10.0])
    m = np.array([5.0, 7.0])
    # Intervalo degenerado (lo = hi = mediana): WIS = |y-m| * (0.5 + alpha/2 * 2/alpha) / 1.5
    expected = np.mean(np.abs(y - m)) * (0.5 + 1.0) / 1.5
    assert np.isclose(wis(y, m, m, m), expected)


def test_diebold_mariano_detects_clear_improvement():
    rng = np.random.default_rng(0)
    d = rng.normal(-1.0, 1.0, size=200)  # el modelo tiene menor perdida
    stat, p = diebold_mariano(d, h=1)
    assert stat < 0
    assert p < 0.001


def test_diebold_mariano_no_difference():
    rng = np.random.default_rng(1)
    d = rng.normal(0.0, 1.0, size=200)
    _, p = diebold_mariano(d, h=1)
    assert p > 0.01
