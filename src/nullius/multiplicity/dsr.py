"""Deflated Sharpe Ratio — Bailey & López de Prado (2014), *J. Portfolio Management* 40(5).

A raw Sharpe selected as the best of many trials is biased upward: with N trials, even
pure noise produces an expected maximum Sharpe well above zero. The DSR asks whether the
selected strategy's Sharpe exceeds *that* expected maximum, given the number of trials,
sample length, skew and kurtosis.

``deflated_sharpe_ratio``, ``_psr`` and ``trial_sharpe_variance`` are ported verbatim
from the source repo (golden-parity tested). ``deflated_sharpe`` is a thin
:class:`CheckResult` wrapper with registry sugar.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import norm
from scipy.stats import skew as _skew

from ..clock import DAILY, Clock
from ..result import CheckResult

_EULER_GAMMA = 0.5772156649015329


def trial_sharpe_variance(
    trial_sharpes: Iterable[float],
    *,
    annualized: bool = True,
    periods_per_year: float = 252,
) -> float:
    """Variance of the trial Sharpes in PER-PERIOD units.

    The DSR formula uses a per-period ``sr_hat``; the cross-trial variance ``V`` must be
    in the same units. Trial registries typically store ANNUALIZED Sharpes, so pass
    ``annualized=True`` (default) to divide by ``sqrt(periods_per_year)`` before taking
    the variance. Computing ``V`` directly from annualized numbers inflates it
    ~``periods_per_year``× and makes DSR far too lenient.
    """
    s = np.asarray(list(trial_sharpes), dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 2:
        return 0.0
    if annualized:
        s = s / np.sqrt(float(periods_per_year))
    return float(np.var(s, ddof=1))


def _psr(sr_hat: float, sr_ref: float, T: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe Ratio: P(true per-period Sharpe > sr_ref).

    ``kurt`` is the non-excess kurtosis (normal == 3). Returns NaN if the variance term
    is non-positive (never square-roots a non-positive number) or T < 2.
    """
    denom = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat**2
    if denom <= 0 or T < 2:
        return float("nan")
    z = (sr_hat - sr_ref) * np.sqrt(T - 1.0) / np.sqrt(denom)
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    returns: Iterable[float],
    *,
    n_trials: int,
    var_across_trials: float,
    sr_benchmark: float = 0.0,
    periods_per_year: float = 252,
) -> dict:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    Parameters
    ----------
    returns:
        Per-period returns of the SELECTED strategy.
    n_trials:
        Number of trials the selection searched over (N). Drives the expected-max-Sharpe
        haircut — the dominant input; report a band of N.
    var_across_trials:
        Variance of the trials' Sharpes in PER-PERIOD units (use
        :func:`trial_sharpe_variance` to convert from annualized).
    sr_benchmark:
        Per-period Sharpe threshold under H0 (default 0).

    Returns
    -------
    dict
        Keys: ``dsr``, ``psr_vs_benchmark``, ``sr_hat`` (per-period), ``sr_hat_annual``,
        ``sr0`` (expected max under N trials), ``skew``, ``kurt`` (non-excess),
        ``n_trials``, ``var_across_trials``, ``T``, ``passed`` (``dsr >= 0.95``).
    """
    r = pd.Series(list(returns), dtype=float).dropna()
    T = int(len(r))
    if T < 2:
        return {
            "dsr": float("nan"),
            "psr_vs_benchmark": float("nan"),
            "sr_hat": 0.0,
            "sr_hat_annual": 0.0,
            "sr0": 0.0,
            "skew": 0.0,
            "kurt": 3.0,
            "n_trials": int(n_trials),
            "var_across_trials": float(var_across_trials),
            "T": T,
            "passed": False,
        }

    sd = float(r.std(ddof=1))
    sr_hat = float(r.mean() / sd) if sd > 0 else 0.0
    skew = float(_skew(r))
    kurt = float(_kurtosis(r, fisher=False))  # normal == 3.0 (NOT excess)

    N = max(int(n_trials), 1)
    V = float(var_across_trials)
    if N <= 1 or V <= 0:
        # No multiple-testing haircut with a single trial (or no cross-trial spread).
        # Guards against Phi^-1(0) = -inf at N=1, which would make sr0 undefined.
        sr0 = float(sr_benchmark)
    else:
        emax = (1.0 - _EULER_GAMMA) * norm.ppf(1.0 - 1.0 / N) + _EULER_GAMMA * norm.ppf(
            1.0 - 1.0 / (N * np.e)
        )
        sr0 = float(sr_benchmark + np.sqrt(V) * emax)

    dsr = _psr(sr_hat, sr0, T, skew, kurt)
    psr_vs_benchmark = _psr(sr_hat, float(sr_benchmark), T, skew, kurt)

    return {
        "dsr": dsr,
        "psr_vs_benchmark": psr_vs_benchmark,
        "sr_hat": sr_hat,
        "sr_hat_annual": sr_hat * np.sqrt(float(periods_per_year)),
        "sr0": sr0,
        "skew": skew,
        "kurt": kurt,
        "n_trials": int(N),
        "var_across_trials": V,
        "T": T,
        "passed": bool(np.isfinite(dsr) and dsr >= 0.95),
    }


def deflated_sharpe(
    returns: Iterable[float],
    *,
    n_trials: int | None = None,
    var_across_trials: float | None = None,
    registry=None,
    sharpe_key: str = "sharpe",
    sr_benchmark: float = 0.0,
    clock: Clock = DAILY,
) -> CheckResult:
    """Deflated Sharpe as a :class:`CheckResult`, with optional registry sugar.

    Detects a strategy whose Sharpe looks significant only because you forgot how many
    trials you searched. Pass ``n_trials`` and per-period ``var_across_trials``
    explicitly, or pass a :class:`~nullius.multiplicity.trials.TrialRegistry` and let the
    check read N from it and convert the annualized trial Sharpes (under ``sharpe_key``)
    to per-period variance automatically — closing the units trap.
    """
    if registry is not None:
        if n_trials is None:
            n_trials = registry.n_trials()
        if var_across_trials is None:
            var_across_trials = trial_sharpe_variance(
                registry.metric_values(sharpe_key),
                annualized=True,
                periods_per_year=clock.periods_per_year,
            )
    if n_trials is None or var_across_trials is None:
        raise ValueError(
            "provide n_trials and var_across_trials, or a registry to derive them from"
        )

    d = deflated_sharpe_ratio(
        returns,
        n_trials=n_trials,
        var_across_trials=var_across_trials,
        sr_benchmark=sr_benchmark,
        periods_per_year=clock.periods_per_year,
    )
    passed = bool(d["passed"])
    details = (
        f"DSR={d['dsr']:.4g} vs 0.95 threshold. Annualized Sharpe {d['sr_hat_annual']:.3g} "
        f"over T={d['T']}, deflated for N={d['n_trials']} trial(s) "
        f"(expected-max per-period Sharpe under H0 = {d['sr0']:.4g}). "
        + (
            "The selected Sharpe survives the multiple-testing haircut."
            if passed
            else "The selected Sharpe does NOT survive the multiple-testing haircut — "
            "it is consistent with the best of N noise trials."
        )
    )
    return CheckResult(
        name="deflated_sharpe",
        passed=passed,
        severity="fatal",
        statistic=float(d["dsr"]) if np.isfinite(d["dsr"]) else None,
        threshold=0.95,
        details=details,
    )
