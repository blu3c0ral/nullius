"""Diebold-Mariano test for equal predictive accuracy, with a panel-clustered variant.

Layered API (distinct names, no collision):

- :func:`dm_stat` — the low-level kernel on a *precomputed* loss-differential series
  (Bartlett HAC long-run variance + exact Harvey-Leybourne-Newbold small-sample factor,
  ``t(n-1)`` reference). NEW.
- :func:`diebold_mariano` — the ported convenience function that computes the loss
  differential internally for a chosen loss and calls :func:`dm_stat`. Byte-for-byte
  parity with the source repo.
- :func:`diebold_mariano_panel` — clusters a panel by **time**: averages the loss
  differential across assets per timestamp, then runs :func:`dm_stat` on the per-time
  series (Driscoll-Kraay style). Pooling N correlated assets into one long series inflates
  the DM statistic by up to √N.
- :func:`dm_test` — the public dispatcher returning a :class:`CheckResult`: panel variant
  when an ``asset`` column is present, else the pooled series.

Sign convention: a negative statistic means ``pred1`` has lower loss (is better).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as t_dist

from ..clock import DAILY, Clock
from ..result import CheckResult
from ._common import ArrayLike, default_hac_lag, to_1d_array
from .qlike import qlike_loss


@dataclass
class DMResult:
    """Outcome of the DM kernel on a loss-differential series."""

    dm_stat: float
    dm_p_value: float
    dm_stat_hln: float
    dm_p_value_hln: float
    mean_diff: float
    n_obs: int
    hac_lag_used: int


def dm_stat(loss_diff: ArrayLike, *, hac_lag: int | None = None, horizon: int = 1) -> DMResult:
    """Diebold-Mariano statistic on a precomputed loss differential ``d = L1 - L2``.

    Estimates the long-run variance of ``mean(d)`` with a Bartlett-kernel HAC (manual, so
    it exactly reproduces the source), applies the HLN finite-sample correction and
    reports both the standard-normal and ``t(n-1)`` p-values. NaNs in ``loss_diff`` are
    dropped.
    """
    d = to_1d_array(loss_diff, "loss_diff")
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        raise ValueError(f"dm_stat requires at least 2 non-NaN observations, got {n}")

    L = hac_lag if hac_lag is not None else default_hac_lag(n, horizon)
    mean_diff = float(d.mean())

    if d.std() == 0:
        return DMResult(0.0, 1.0, 0.0, 1.0, mean_diff, n, L)

    d_centered = d - d.mean()
    long_run_var = float((d_centered**2).mean())
    for k in range(1, L + 1):
        gamma_k = float((d_centered[k:] * d_centered[:-k]).mean())
        w_k = 1.0 - k / (L + 1)
        long_run_var += 2 * w_k * gamma_k

    sample_var = float((d_centered**2).mean())
    if long_run_var < sample_var / n:
        long_run_var = sample_var
    var_d_bar = long_run_var / n

    stat = float(mean_diff / np.sqrt(var_d_bar))
    h = horizon
    hln_factor = float(np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n))
    stat_hln = stat * hln_factor
    p_value = float(2.0 * norm.sf(abs(stat)))
    p_value_hln = float(t_dist.sf(abs(stat_hln), df=n - 1) * 2)
    return DMResult(stat, p_value, stat_hln, p_value_hln, mean_diff, n, L)


def diebold_mariano(
    actual: ArrayLike,
    pred1: ArrayLike,
    pred2: ArrayLike,
    *,
    loss: str | Callable = "qlike",
    horizon: int = 1,
    hac_lag: int | None = None,
) -> dict:
    """Diebold-Mariano test for equal predictive accuracy of ``pred1`` vs ``pred2``.

    Detects whether one forecast is genuinely more accurate or merely luckier on the
    sample. ``loss`` is ``"qlike"`` (VARIANCE inputs), ``"mse"``, ``"mae"``, or a callable
    ``loss(actual, pred) -> 1-D array``. Requires at least 30 non-NaN observations.

    Returns a dict with ``dm_stat``, ``dm_p_value``, ``dm_stat_hln``, ``dm_p_value_hln``,
    ``mean_diff``, ``loss1_mean``, ``loss2_mean``, ``n_obs``, ``hac_lag_used``,
    ``loss_name``, ``better_model``.
    """
    a = to_1d_array(actual, "actual")
    p1 = to_1d_array(pred1, "pred1")
    p2 = to_1d_array(pred2, "pred2")

    mask_valid = ~(np.isnan(a) | np.isnan(p1) | np.isnan(p2))
    a_c, p1_c, p2_c = a[mask_valid], p1[mask_valid], p2[mask_valid]
    n = len(a_c)
    if n < 30:
        raise ValueError(f"diebold_mariano requires at least 30 non-NaN observations, got {n}")

    loss_name = "custom" if (callable(loss) and not isinstance(loss, str)) else str(loss)

    if loss == "qlike":
        l1 = qlike_loss(a_c, p1_c).to_numpy()
        l2 = qlike_loss(a_c, p2_c).to_numpy()
    elif loss == "mse":
        l1 = (a_c - p1_c) ** 2
        l2 = (a_c - p2_c) ** 2
    elif loss == "mae":
        l1 = np.abs(a_c - p1_c)
        l2 = np.abs(a_c - p2_c)
    elif callable(loss):
        l1 = np.asarray(loss(a_c, p1_c), dtype=float)
        l2 = np.asarray(loss(a_c, p2_c), dtype=float)
        for name, arr in (("pred1", l1), ("pred2", l2)):
            if arr.ndim != 1 or len(arr) != n:
                raise ValueError(
                    f"Custom loss on {name} must return a 1-D array of length {n}, "
                    f"got shape {arr.shape}."
                )
    else:
        raise ValueError(f"Unknown loss {loss!r}. Use 'qlike', 'mse', 'mae', or a callable.")

    d = l1 - l2
    loss1_mean = float(np.nanmean(l1))
    loss2_mean = float(np.nanmean(l2))
    mean_diff = float(np.nanmean(d))
    L = hac_lag if hac_lag is not None else default_hac_lag(n, horizon)

    clean = d[~np.isnan(d)]
    if clean.std() == 0:
        return {
            "dm_stat": 0.0,
            "dm_p_value": 1.0,
            "dm_stat_hln": 0.0,
            "dm_p_value_hln": 1.0,
            "mean_diff": mean_diff,
            "loss1_mean": loss1_mean,
            "loss2_mean": loss2_mean,
            "n_obs": n,
            "hac_lag_used": L,
            "loss_name": loss_name,
            "better_model": "tie",
        }

    res = dm_stat(clean, hac_lag=L, horizon=horizon)
    return {
        "dm_stat": res.dm_stat,
        "dm_p_value": res.dm_p_value,
        "dm_stat_hln": res.dm_stat_hln,
        "dm_p_value_hln": res.dm_p_value_hln,
        "mean_diff": mean_diff,
        "loss1_mean": loss1_mean,
        "loss2_mean": loss2_mean,
        "n_obs": n,
        "hac_lag_used": L,
        "loss_name": loss_name,
        "better_model": "pred1" if mean_diff < 0 else "pred2",
    }


def diebold_mariano_panel(
    losses: pd.DataFrame,
    *,
    time_col: str = "time",
    asset_col: str = "asset",
    loss_diff_col: str = "loss_diff",
    hac_lag: int | None = None,
    horizon: int = 1,
) -> DMResult:
    """DM on a panel, clustered by time (Driscoll-Kraay style).

    Averages the per-row loss differential across assets within each timestamp, then runs
    :func:`dm_stat` on the resulting per-time series. This avoids the √N inflation that
    pooling N correlated assets into one long series would produce.
    """
    for col in (time_col, loss_diff_col):
        if col not in losses.columns:
            raise ValueError(f"losses frame missing column {col!r}")
    per_time = losses.groupby(time_col)[loss_diff_col].mean().sort_index().to_numpy(dtype=float)
    return dm_stat(per_time, hac_lag=hac_lag, horizon=horizon)


def dm_test(
    losses: pd.DataFrame,
    *,
    time_col: str = "time",
    asset_col: str = "asset",
    loss_diff_col: str = "loss_diff",
    hac_lag: int | None = None,
    horizon: int = 1,
    alpha: float = 0.05,
    cluster: bool | None = None,
    clock: Clock = DAILY,
) -> CheckResult:
    """Public DM dispatcher returning an informational :class:`CheckResult`.

    Pass a frame with a ``loss_diff`` column (``L1 - L2`` per row). If an ``asset`` column
    is present the panel/time-clustered variant runs by default; forcing ``cluster=False``
    on panel-shaped data runs the pooled series and raises the severity to ``warning``,
    because pooling correlated assets inflates the DM statistic by up to √N.

    DM is a two-sided test of *equal* predictive accuracy, not a pass/fail guardrail, so the
    result is informational (``passed is None``); whether the difference is significant at
    ``alpha`` (and which model it favours) is reported in ``details``. For overlapping
    multi-bar forecasts set ``horizon`` (in bars) so the HAC lag widens accordingly, or fix
    it via ``clock.hac_lag``; the per-time series of the panel variant is autocorrelated at
    the forecast horizon.
    """
    if loss_diff_col not in losses.columns:
        raise ValueError(f"losses frame missing column {loss_diff_col!r}")
    effective_lag = hac_lag if hac_lag is not None else clock.hac_lag
    has_asset = asset_col in losses.columns
    use_panel = has_asset if cluster is None else (cluster and has_asset)
    forced_pool = has_asset and cluster is False

    if use_panel:
        res = diebold_mariano_panel(
            losses,
            time_col=time_col,
            asset_col=asset_col,
            loss_diff_col=loss_diff_col,
            hac_lag=effective_lag,
            horizon=horizon,
        )
        method = "panel (clustered by time)"
    else:
        res = dm_stat(
            losses[loss_diff_col].to_numpy(dtype=float), hac_lag=effective_lag, horizon=horizon
        )
        method = "pooled series"

    significant = res.dm_p_value_hln < alpha
    severity = "warning" if forced_pool else "info"
    favour = "model 1" if res.mean_diff < 0 else "model 2"
    details = (
        f"DM ({method}): stat={res.dm_stat:.3g}, HLN stat={res.dm_stat_hln:.3g}, "
        f"HLN p={res.dm_p_value_hln:.3g} over n={res.n_obs} (HAC lag {res.hac_lag_used}). "
        + (
            f"Accuracy difference significant at alpha={alpha}, favouring {favour}."
            if significant
            else f"No significant accuracy difference at alpha={alpha}."
        )
        + (
            " WARNING: pooled series forced on panel-shaped data — the statistic is "
            "inflated (up to √N); prefer the default time-clustered variant."
            if forced_pool
            else ""
        )
    )
    # Threshold on the same reference distribution used for `significant` (t, df=n-1).
    crit = float(t_dist.isf(alpha / 2.0, df=max(res.n_obs - 1, 1)))
    return CheckResult(
        name="diebold_mariano",
        passed=None,
        severity=severity,
        statistic=res.dm_stat_hln,
        threshold=crit,
        details=details,
    )
