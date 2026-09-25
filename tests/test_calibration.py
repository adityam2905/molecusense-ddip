"""Calibration maths and the per-drug additive fit used by the pair audit."""

import numpy as np
import pytest

from utils.calibration import (
    fit_temperature, expected_calibration_error, brier_score, percentile, band_rates,
)
from experiments.pair_audit import additive_fit


def test_temperature_recovers_known_value():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3, 20000)                      # true log-odds
    y = (rng.random(z.size) < 1 / (1 + np.exp(-z))).astype(float)
    overconfident = 2.0 * z                          # needs T = 2 to undo
    assert fit_temperature(overconfident, y) == pytest.approx(2.0, rel=0.05)


def test_ece_and_brier():
    rng = np.random.default_rng(1)
    p = rng.random(50000)
    y = (rng.random(p.size) < p).astype(float)
    assert expected_calibration_error(p, y) < 0.01          # calibrated
    assert expected_calibration_error(np.full_like(p, 0.99), y) > 0.4
    assert brier_score(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0


def test_percentile_and_bands():
    ref = np.arange(100, dtype=float)                # 100 non-interacting scores
    assert percentile(-1.0, ref) == 0.0
    assert percentile(95.0, ref) == 95.0
    rates = band_rates(ref, np.zeros(100), ref)
    assert rates["non_interacting"]["HIGH"] == pytest.approx(0.05)


def test_additive_fit_matches_least_squares():
    rng = np.random.default_rng(2)
    n = 25
    m = rng.normal(size=(n, n)); m = (m + m.T) / 2
    _, r2 = additive_fit(m)

    iu = np.triu_indices(n, 1)
    X = np.zeros((len(iu[0]), n + 1)); X[:, 0] = 1
    X[np.arange(len(iu[0])), iu[0] + 1] = 1
    X[np.arange(len(iu[0])), iu[1] + 1] = 1
    coef, *_ = np.linalg.lstsq(X, m[iu], rcond=None)
    resid = m[iu] - X @ coef
    r2_ls = 1 - (resid ** 2).sum() / ((m[iu] - m[iu].mean()) ** 2).sum()
    assert r2 == pytest.approx(r2_ls, abs=1e-9)


def test_additive_fit_is_perfect_on_per_drug_scores():
    s = np.random.default_rng(3).normal(size=12)
    m = 0.5 + s[:, None] + s[None, :]
    _, r2 = additive_fit(m)
    assert r2 == pytest.approx(1.0)
