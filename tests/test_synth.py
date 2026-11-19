"""Ridge-augmented synthetic control: simulation tests with a known effect."""

import numpy as np
import pandas as pd
import pytest

from caz import synth

T, T0, J = 74, 21, 15  # weeks, pre-treatment weeks, donors: as in the Birmingham replication


def factor_panel(effect: float, seed: int, noise: float = 0.5):
    """Weekly series from a 3-factor model (seasonal, trend, noise-like), effect after T0.

    Donors and treated unit load on the same factors, so a good synthetic
    control exists; the treated unit's loadings are not a convex combination of
    the donors', so plain SCM is slightly biased and the ridge correction matters.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    factors = np.vstack([
        np.cos(2 * np.pi * t / 52),  # seasonal
        t / T,  # trend
        np.cumsum(rng.normal(0, 0.3, T)),  # shared shocks
    ])  # fmt: skip
    loadings = rng.uniform(0.5, 1.5, (J, 3))
    y0 = 30 + loadings @ factors * 8 + rng.normal(0, noise, (J, T))
    treated_loadings = loadings.mean(axis=0) * 1.1  # just outside the donors' hull
    y1 = 30 + treated_loadings @ factors * 8 + rng.normal(0, noise, T)
    y1[T0:] += effect
    return y1, y0


def test_scm_weights_are_a_convex_combination() -> None:
    y1, y0 = factor_panel(effect=0, seed=0)
    w = synth.scm_weights(y1[:T0], y0[:, :T0])
    assert w.min() >= 0 and w.sum() == pytest.approx(1)


def test_scm_recovers_exact_weights_when_the_treated_unit_is_a_mix_of_donors() -> None:
    rng = np.random.default_rng(3)
    x0 = rng.normal(0, 1, (5, 30))
    true = np.array([0.6, 0.4, 0, 0, 0])
    assert synth.scm_weights(true @ x0, x0) == pytest.approx(true, abs=1e-4)


def test_ridge_weights_with_huge_lambda_equal_scm_weights() -> None:
    y1, y0 = factor_panel(effect=0, seed=1)
    w_scm = synth.scm_weights(y1[:T0], y0[:, :T0])
    w_aug = synth.ridge_augmented_weights(y1[:T0], y0[:, :T0], lam=1e12)
    assert w_aug == pytest.approx(w_scm, abs=1e-6)


@pytest.mark.parametrize("effect", [-3.0, 0.0, 2.0])
def test_known_effect_is_recovered(effect: float) -> None:
    errors = []
    for seed in range(10):
        y1, y0 = factor_panel(effect, seed)
        result = synth.fit(y1, y0, T0)
        window = np.arange(T) >= T0
        errors.append(synth.average_effect(result, y1, window).absolute - effect)
    assert abs(np.mean(errors)) < 0.5  # unbiased on average
    assert np.sqrt(np.mean(np.square(errors))) < 1.5  # and close in each run


def test_ridge_correction_reduces_pre_treatment_imbalance() -> None:
    y1, y0 = factor_panel(effect=0, seed=4)
    result = synth.fit(y1, y0, T0)
    scm_gap = y1[:T0] - result.scm_weights @ y0[:, :T0]
    assert np.sqrt(np.mean(result.gap[:T0] ** 2)) < np.sqrt(np.mean(scm_gap**2))


def test_intervals_cover_the_true_effect_most_of_the_time() -> None:
    covered = []
    for seed in range(40):
        y1, y0 = factor_panel(-3.0, seed + 100)
        result = synth.fit(y1, y0, T0)
        e = synth.average_effect(result, y1, np.arange(T) >= T0)
        covered.append(e.lower <= -3.0 <= e.upper)
    assert np.mean(covered) >= 0.8  # nominal 95%; jackknife+ guarantees at least ~90% - 2 alpha


def test_percent_change_is_relative_to_the_synthetic_level() -> None:
    y1, y0 = factor_panel(-3.0, seed=7)
    result = synth.fit(y1, y0, T0)
    window = np.arange(T) >= T0
    e = synth.average_effect(result, y1, window)
    assert e.percent == pytest.approx(100 * e.absolute / result.synthetic[window].mean())


def test_missing_values_are_refused() -> None:
    y1, y0 = factor_panel(0, seed=0)
    y0[2, 5] = np.nan
    with pytest.raises(ValueError, match="missing"):
        synth.fit(y1, y0, T0)


def test_short_donor_gaps_are_filled_and_long_ones_drop_the_donor() -> None:
    nan = float("nan")
    panel = pd.DataFrame({
        "short": [1, nan, nan, 4, 5, 6],  # interior gap of 2: interpolated
        "edge": [nan, 2, 3, 4, 5, 6],  # first value missing: takes the next
        "long": [1, nan, nan, nan, nan, nan],  # trailing gap of 5: dropped
        "long_inside": [1, nan, nan, nan, nan, 6],  # interior gap 4 > 3: dropped
        "edge_two": [1, 2, 3, 4, nan, nan],  # trailing gap of 2 > 1: dropped
    })  # fmt: skip
    filled, dropped = synth.fill_donor_gaps(panel, max_gap=3, max_edge_gap=1)
    assert filled["short"].tolist() == [1, 2, 3, 4, 5, 6]
    assert filled["edge"].tolist() == [2, 2, 3, 4, 5, 6]
    assert sorted(dropped) == ["edge_two", "long", "long_inside"]
