"""Tests for the telemetry noise model."""

from __future__ import annotations

import numpy as np
import pytest

from sentinelops.sim.telemetry import NoiseConfig, add_noise, reading


def test_noise_is_reproducible_and_keeps_ranges() -> None:
    amounts = np.array([[100.0, 0.0, np.nan]])
    fractions = np.array([[0.0, 1.0, 0.5, np.nan]])
    cfg = NoiseConfig(relative_cv=0.5, fraction_sd=0.5, missing_prob=0.0)
    a1, f1 = add_noise(np.random.default_rng(1), amounts, fractions, cfg)
    a2, f2 = add_noise(np.random.default_rng(1), amounts, fractions, cfg)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(f1, f2)
    assert (a1[~np.isnan(a1)] >= 0).all()
    assert ((f1[~np.isnan(f1)] >= 0) & (f1[~np.isnan(f1)] <= 1)).all()
    assert a1[0, 1] == 0.0  # zero stays zero under multiplicative noise
    assert np.isnan(a1[0, 2]) and np.isnan(f1[0, 3])  # unmeasurable stays missing


def test_noise_draws_do_not_depend_on_values() -> None:
    cfg = NoiseConfig()
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    add_noise(rng_a, np.zeros((3, 2)), np.zeros((3, 1)), cfg)
    add_noise(rng_b, np.full((3, 2), 1e6), np.full((3, 1), np.nan), cfg)
    assert rng_a.random() == rng_b.random()


def test_zero_noise_is_identity_and_full_missing_hides_all() -> None:
    amounts, fractions = np.array([[3.0, 4.0]]), np.array([[0.25]])
    a, f = add_noise(np.random.default_rng(0), amounts, fractions, NoiseConfig(0.0, 0.0, 0.0))
    np.testing.assert_array_equal(a, amounts)
    np.testing.assert_array_equal(f, fractions)
    a, f = add_noise(np.random.default_rng(0), amounts, fractions, NoiseConfig(missing_prob=1.0))
    assert np.isnan(a).all() and np.isnan(f).all()


def test_noise_level_matches_config() -> None:
    a, _ = add_noise(
        np.random.default_rng(3),
        np.full((50_000, 1), 100.0),
        np.zeros((1, 1)),
        NoiseConfig(relative_cv=0.05, missing_prob=0.0),
    )
    assert np.std(a) == pytest.approx(5.0, rel=0.05)


@pytest.mark.parametrize("kwargs", [{"relative_cv": -1}, {"fraction_sd": -1}, {"missing_prob": 2}])
def test_noise_config_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        NoiseConfig(**kwargs)


def test_reading_converts_nan_to_none() -> None:
    assert reading(float("nan")) is None
    assert reading(np.float64(2.5)) == 2.5
