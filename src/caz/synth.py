"""Ridge-augmented synthetic control (Ben-Michael, Feller & Rothstein 2021, JASA).

One treated unit, J donors, T periods of which the first T0 are before treatment.
1. Synthetic control (SCM) weights w: non-negative, summing to 1, chosen so the
   weighted donors best match the treated unit before treatment.
2. Ridge augmentation: SCM rarely matches exactly. A ridge regression across
   donors, from pre-treatment outcomes, corrects the remaining imbalance:
       w_aug = w + (X1 - X0' w)' (X0c' X0c + lambda I)^-1 X0c'
   where X0c are the donors' pre-treatment outcomes centred across donors.
   The weights may then be slightly negative (mild extrapolation).
3. The counterfactual for the treated unit is sum_j w_aug_j Y_jt; the effect at
   time t is Y_1t minus it.
lambda is chosen by rolling-origin cross-validation over the last pre-treatment
periods. Intervals use the time jackknife+ (as in augsynth's "jackknife+"): refit
leaving out one pre-treatment period at a time, and widen each refit's effect
estimate by its error on the period it left out.

Written here rather than imported: the R package augsynth (used by Liu et al.
2023) has no maintained Python equivalent; see docs/DECISIONS.md.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

N_CV_PERIODS = 5  # last pre-treatment periods used to choose lambda
LAMBDA_GRID_SIZE = 20


def scm_weights(x1: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """SCM weights: argmin ||x1 - x0' w||^2 subject to w >= 0, sum(w) = 1.

    x1: (T0,) treated pre-treatment outcomes; x0: (J, T0) donors'.
    """
    n_donors = x0.shape[0]

    def loss(w: np.ndarray) -> float:
        r = x1 - x0.T @ w
        return float(r @ r)

    def grad(w: np.ndarray) -> np.ndarray:
        return -2 * x0 @ (x1 - x0.T @ w)

    result = minimize(
        loss,
        np.full(n_donors, 1 / n_donors),
        jac=grad,
        bounds=[(0, 1)] * n_donors,
        constraints=[
            {"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)}
        ],
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"SCM weights did not converge: {result.message}")
    w = np.clip(result.x, 0, None)
    return w / w.sum()


def ridge_augmented_weights(x1: np.ndarray, x0: np.ndarray, lam: float) -> np.ndarray:
    """Ridge ASCM weights (see module docstring); x1: (T0,), x0: (J, T0)."""
    w = scm_weights(x1, x0)
    x0c = x0 - x0.mean(axis=0)  # centre each period across donors
    x1c = x1 - x0.mean(axis=0)
    gram = x0c.T @ x0c + lam * np.eye(x0.shape[1])
    imbalance = x1c - x0c.T @ w
    return w + x0c @ np.linalg.solve(gram, imbalance)


def lambda_grid(x0: np.ndarray) -> np.ndarray:
    """Candidate ridge penalties, scaled to the donors' pre-treatment variation."""
    x0c = x0 - x0.mean(axis=0)
    top = np.linalg.eigvalsh(x0c.T @ x0c).max()
    return top * np.geomspace(1e-4, 10, LAMBDA_GRID_SIZE)


def choose_lambda(x1: np.ndarray, x0: np.ndarray) -> float:
    """lambda with the smallest error predicting each of the last N_CV_PERIODS
    pre-treatment periods from the periods before it."""
    t0 = len(x1)
    holdouts = range(max(t0 - N_CV_PERIODS, 2), t0)
    grid = lambda_grid(x0)
    errors = []
    for lam in grid:
        sq = [(x1[k] - ridge_augmented_weights(x1[:k], x0[:, :k], lam) @ x0[:, k]) ** 2
              for k in holdouts]  # fmt: skip
        errors.append(np.mean(sq))
    return float(grid[int(np.argmin(errors))])


@dataclass
class SynthResult:
    weights: np.ndarray  # (J,) ridge-augmented donor weights
    scm_weights: np.ndarray  # (J,) plain SCM weights, for reporting
    lam: float
    synthetic: np.ndarray  # (T,) counterfactual for the treated unit
    gap: np.ndarray  # (T,) treated minus synthetic
    loo_synthetic: np.ndarray  # (T0, T) counterfactuals refitted without each pre period
    loo_residual: np.ndarray  # (T0,) |error| on the left-out pre period
    t0: int


def fit(y1: np.ndarray, y0: np.ndarray, t0: int, lam: float | None = None) -> SynthResult:
    """Ridge ASCM for treated series y1 (T,) and donors y0 (J, T); the first t0 are pre."""
    x1, x0 = y1[:t0], y0[:, :t0]
    if np.isnan(y1).any() or np.isnan(y0).any():
        raise ValueError("series must have no missing values (impute or drop units first)")
    lam = choose_lambda(x1, x0) if lam is None else lam
    weights = ridge_augmented_weights(x1, x0, lam)
    synthetic = weights @ y0

    loo_synthetic = np.empty((t0, len(y1)))
    loo_residual = np.empty(t0)
    for t in range(t0):  # time jackknife+: leave one pre period out
        keep = np.arange(t0) != t
        w_t = ridge_augmented_weights(x1[keep], x0[:, keep], lam)
        loo_synthetic[t] = w_t @ y0
        loo_residual[t] = abs(y1[t] - loo_synthetic[t, t])
    return SynthResult(
        weights=weights,
        scm_weights=scm_weights(x1, x0),
        lam=lam,
        synthetic=synthetic,
        gap=y1 - synthetic,
        loo_synthetic=loo_synthetic,
        loo_residual=loo_residual,
        t0=t0,
    )


@dataclass
class Effect:
    absolute: float  # mean of treated - synthetic over the window (units of y)
    lower: float  # 95% jackknife+ interval
    upper: float
    percent: float  # absolute / mean synthetic * 100, as in Liu et al. (2023)
    percent_lower: float
    percent_upper: float


def average_effect(result: SynthResult, y1: np.ndarray, window: np.ndarray, alpha: float = 0.05):
    """Average effect over the periods selected by boolean `window` (post-treatment)."""
    effect = float(result.gap[window].mean())
    base = float(result.synthetic[window].mean())
    # each leave-one-out refit gives an effect; widen it by that refit's held-out error
    loo_effects = (y1[window] - result.loo_synthetic[:, window]).mean(axis=1)
    lower = float(np.quantile(loo_effects - result.loo_residual, alpha / 2))
    upper = float(np.quantile(loo_effects + result.loo_residual, 1 - alpha / 2))
    return Effect(
        absolute=effect,
        lower=lower,
        upper=upper,
        percent=100 * effect / base,
        percent_lower=100 * lower / base,
        percent_upper=100 * upper / base,
    )


def fill_donor_gaps(
    panel: pd.DataFrame, max_gap: int = 4, max_edge_gap: int = 1
) -> tuple[pd.DataFrame, list[str]]:
    """Fill short gaps in donor series (periods x donors); drop donors with longer ones.

    Interior gaps of at most `max_gap` periods are filled by linear interpolation;
    a gap at the start or end of at most `max_edge_gap` periods takes the nearest
    value. A donor with any longer gap is dropped (not partly filled). Returns
    (filled panel of the kept donors, names of dropped donors).
    """
    kept, dropped = {}, []
    for name, series in panel.items():
        missing = series.isna().to_numpy()
        runs = _missing_runs(missing)
        n = len(series)
        too_long = any(
            (length > max_edge_gap) if (start == 0 or start + length == n) else (length > max_gap)
            for start, length in runs
        )
        if too_long or missing.all():
            dropped.append(name)
            continue
        kept[name] = series.interpolate(limit_area="inside").ffill().bfill()
    return pd.DataFrame(kept, index=panel.index), dropped


def _missing_runs(missing: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of each run of True values."""
    runs, start = [], None
    for i, m in enumerate(missing):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(missing) - start))
    return runs
