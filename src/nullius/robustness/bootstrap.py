"""Stationary-bootstrap IC p-value (Politis & Romano 1994) + Fisher combination.

Standard IC p-values assume iid observations; with autocorrelated targets (overlapping
forward returns) they are invalid. The stationary bootstrap resamples in geometric-length
blocks, preserving short-range dependence while breaking the prediction/outcome alignment
under H0 = "no predictive relationship". Ported verbatim from the source repo.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    r, _ = spearmanr(x, y)
    if np.isnan(r):
        return 0.0
    return float(r)


def _stationary_bootstrap_indices(
    n: int, mean_block_length: float, rng: np.random.Generator
) -> np.ndarray:
    """Length-n bootstrap indices using Politis-Romano geometric blocks.

    At each step, with probability ``1/mean_block_length`` start a new block at a random
    index; otherwise continue the current block by incrementing (wrapping around).
    """
    p = 1.0 / float(mean_block_length)
    out = np.empty(n, dtype=np.int64)
    out[0] = rng.integers(0, n)
    for i in range(1, n):
        if rng.random() < p:
            out[i] = rng.integers(0, n)
        else:
            out[i] = (out[i - 1] + 1) % n
    return out


def ic_bootstrap_pvalue(
    predictions: pd.Series,
    actuals: pd.Series,
    *,
    block_length: float = 30.0,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Mapping[str, float]:
    """Stationary-bootstrap p-value for the Spearman information coefficient.

    ``predictions`` and ``actuals`` must be aligned (same index). Resamples PREDICTIONS
    only (keeping actuals fixed), preserving predictions' autocorrelation while breaking
    the prediction/outcome alignment — the null of independent predictions. Two-sided
    p-value uses the unbiased Monte-Carlo convention ``(#|boot| >= |obs| + 1)/(B + 1)``.

    Set ``block_length`` >= the forward-return window so blocks span at least one full
    overlapping window. Use 5000-10000 bootstraps for p-values near alpha.

    Returns ``observed_ic``, ``bootstrap_ic_mean``, ``bootstrap_ic_std``, ``p_value``,
    ``ci_95_low``, ``ci_95_high``, ``n``.
    """
    p = predictions.dropna().astype(float)
    a = actuals.loc[p.index].dropna().astype(float)
    common = p.index.intersection(a.index)
    p = p.loc[common].values
    a = a.loc[common].values
    n = len(p)
    if n < 2:
        return {
            "observed_ic": 0.0,
            "bootstrap_ic_mean": 0.0,
            "bootstrap_ic_std": 0.0,
            "p_value": 1.0,
            "ci_95_low": 0.0,
            "ci_95_high": 0.0,
            "n": n,
        }

    observed = _spearman(p, a)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = _stationary_bootstrap_indices(n, block_length, rng)
        boot[i] = _spearman(p[idx], a)

    p_two_sided = float((np.sum(np.abs(boot) >= abs(observed)) + 1) / (n_bootstrap + 1))
    return {
        "observed_ic": float(observed),
        "bootstrap_ic_mean": float(np.mean(boot)),
        "bootstrap_ic_std": float(np.std(boot, ddof=1)),
        "p_value": p_two_sided,
        "ci_95_low": float(np.percentile(boot, 2.5)),
        "ci_95_high": float(np.percentile(boot, 97.5)),
        "n": int(n),
    }


def fisher_combined_pvalue(per_asset_p_values: pd.Series) -> dict:
    """Fisher's combined probability test across per-asset p-values.

    Combines independent per-asset p-values into a joint test of "no asset has a real
    signal": ``chi2 = -2 * sum(log p_i)``, ``df = 2k``, ``combined_p = 1 - chi2.cdf``.

    Caveat: Fisher's test assumes the p-values are INDEPENDENT. For assets sharing
    cross-sectional drivers this is optimistic — treat it as a screen, not a strict test.
    """
    s = pd.Series(per_asset_p_values).astype(float).dropna()
    k = int(len(s))
    if k == 0:
        return {"chi2": 0.0, "df": 0, "combined_p": 1.0, "n_assets": 0}
    clipped = s.clip(lower=np.finfo(float).tiny)
    chi2_stat = float(-2.0 * np.sum(np.log(clipped.values)))
    df = 2 * k
    combined_p = float(1.0 - chi2.cdf(chi2_stat, df))
    return {"chi2": chi2_stat, "df": df, "combined_p": combined_p, "n_assets": k}
