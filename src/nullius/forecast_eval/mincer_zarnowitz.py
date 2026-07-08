"""Mincer-Zarnowitz regression: is a forecast unbiased and efficient?

Regress realised on predicted, ``actual = alpha + beta * predicted + u``, and jointly test
``H0: alpha = 0, beta = 1`` with Newey-West HAC standard errors. Rejection means the
forecast is miscalibrated in level (``alpha != 0``) and/or slope (``beta != 1``).

Implemented in numpy/scipy so the core has no ``statsmodels`` dependency; the Newey-West
HAC + small-sample correction reproduce ``statsmodels`` OLS(cov_type="HAC",
use_correction=True), which the optional cross-check test verifies.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from ._common import ArrayLike, default_hac_lag, to_1d_array


def _bartlett_hac(X: np.ndarray, resid: np.ndarray, xtx_inv: np.ndarray, L: int) -> np.ndarray:
    """Newey-West HAC covariance of the OLS coefficients (use_correction=True)."""
    n, k = X.shape
    g = X * resid[:, None]  # moment contributions X_t * u_t
    S = g.T @ g  # Gamma_0 (summed, not divided by n)
    for lag in range(1, L + 1):
        gamma = g[lag:].T @ g[:-lag]
        w = 1.0 - lag / (L + 1)
        S += w * (gamma + gamma.T)
    cov = xtx_inv @ S @ xtx_inv
    return cov * (n / (n - k))


def mincer_zarnowitz(
    actual: ArrayLike,
    predicted: ArrayLike,
    *,
    hac_lag: int | None = None,
    horizon: int = 1,
) -> dict:
    """Mincer-Zarnowitz regression with a joint HAC Wald test of ``(alpha, beta) = (0, 1)``.

    Parameters
    ----------
    hac_lag:
        HAC truncation lag; ``None`` uses the Newey-West plug-in default, ``0`` uses
        non-robust OLS standard errors.
    horizon:
        Forecast horizon, used only for the default HAC lag.

    Returns
    -------
    dict
        ``alpha``, ``beta``, ``alpha_se``, ``beta_se``, ``alpha_t``, ``beta_minus_1_t``,
        ``wald_stat``, ``wald_p_value``, ``r_squared``, ``n_obs``, ``hac_lag_used``.
    """
    a = to_1d_array(actual, "actual")
    p = to_1d_array(predicted, "predicted")
    mask = ~(np.isnan(a) | np.isnan(p))
    a_c, p_c = a[mask], p[mask]
    n = len(a_c)
    if n < 30:
        raise ValueError(f"mincer_zarnowitz requires at least 30 non-NaN observations, got {n}.")

    p_range = float(np.max(p_c) - np.min(p_c))
    p_scale = float(np.abs(np.mean(p_c))) if np.mean(p_c) != 0 else 1.0
    if p_range / max(p_scale, 1e-300) < 1e-10:
        raise ValueError("mincer_zarnowitz: predicted has zero variance (constant predictor).")

    L = hac_lag if hac_lag is not None else default_hac_lag(n, horizon)
    k = 2
    X = np.column_stack([np.ones(n), p_c])
    xtx = X.T @ X
    xtx_inv = np.linalg.inv(xtx)
    beta_hat = xtx_inv @ (X.T @ a_c)
    resid = a_c - X @ beta_hat
    alpha, beta = float(beta_hat[0]), float(beta_hat[1])

    rss = float(resid @ resid)
    tss = float(((a_c - a_c.mean()) ** 2).sum())
    r_squared = 1.0 - rss / tss if tss > 0 else 0.0

    if L == 0:
        s2 = rss / (n - k)
        cov = s2 * xtx_inv
    else:
        cov = _bartlett_hac(X, resid, xtx_inv, L)

    alpha_se = float(np.sqrt(cov[0, 0]))
    beta_se = float(np.sqrt(cov[1, 1]))
    alpha_t = alpha / alpha_se
    beta_minus_1_t = (beta - 1.0) / beta_se

    diff = np.array([alpha - 0.0, beta - 1.0])
    resid_std = float(np.std(resid))
    actual_std = float(np.std(a_c)) if np.std(a_c) > 0 else 1.0
    if resid_std / actual_std < 1e-10:
        if abs(alpha) + abs(beta - 1.0) < 1e-10:
            wald_stat, wald_p_value = 0.0, 1.0
        else:
            s2 = rss / (n - k)
            cov_nr = s2 * xtx_inv
            wald_stat = float(diff @ np.linalg.inv(cov_nr) @ diff)
            wald_p_value = float(chi2.sf(wald_stat, k))
    else:
        wald_stat = float(diff @ np.linalg.inv(cov) @ diff)
        wald_p_value = float(chi2.sf(wald_stat, k))

    return {
        "alpha": alpha,
        "beta": beta,
        "alpha_se": alpha_se,
        "beta_se": beta_se,
        "alpha_t": alpha_t,
        "beta_minus_1_t": beta_minus_1_t,
        "wald_stat": wald_stat,
        "wald_p_value": wald_p_value,
        "r_squared": float(r_squared),
        "n_obs": n,
        "hac_lag_used": L,
    }
