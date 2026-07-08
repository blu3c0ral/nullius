"""Inference for Sharpe ratios: standard error and the difference test.

A Sharpe printed without a standard error invites over-reading; a ΔSharpe vs a benchmark
asserted without a test invites over-claiming. Both are answered here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..clock import DAILY, Clock
from ..result import CheckResult


def _per_period_sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else 0.0


def sharpe_se(
    returns,
    *,
    clock: Clock = DAILY,
    annualized: bool = True,
) -> float:
    """Lo (2002) iid standard error of the Sharpe ratio.

    Units trap this function exists to avoid: the formula
    ``SE(SR) = sqrt((1 + SR^2/2) / n)`` uses the **per-period** Sharpe and ``n`` =
    observation count, *not* an annualized Sharpe and a year count. This computes on the
    per-period SR and n, then (if ``annualized``) scales by ``clock.sharpe_factor()`` so
    the SE matches an annualized Sharpe figure.

    The iid caveat: autocorrelation inflates the true SE, so this is a floor. The
    autocorrelation-adjusted variant lives in the intraday extra.
    """
    r = pd.Series(list(returns), dtype=float).dropna().to_numpy()
    n = r.size
    if n < 2:
        return float("nan")
    sr = _per_period_sharpe(r)
    se_pp = np.sqrt((1.0 + 0.5 * sr**2) / n)
    return float(se_pp * clock.sharpe_factor()) if annualized else float(se_pp)


def _align(returns_a, returns_b) -> tuple[np.ndarray, np.ndarray]:
    a = pd.Series(list(returns_a), dtype=float)
    b = pd.Series(list(returns_b), dtype=float)
    if len(a) != len(b):
        raise ValueError(
            f"returns_a and returns_b must have equal length, got {len(a)} and {len(b)}"
        )
    mask = a.notna().to_numpy() & b.notna().to_numpy()
    return a.to_numpy()[mask], b.to_numpy()[mask]


def _circular_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([(np.arange(s, s + block_length) % n) for s in starts])
    return idx[:n]


def sharpe_difference_test(
    returns_a,
    returns_b,
    *,
    clock: Clock = DAILY,
    method: str = "jk_memmel",
    alpha: float = 0.05,
    block_length: int | None = None,
    n_boot: int = 2000,
    seed: int = 42,
    name: str = "sharpe_difference",
) -> CheckResult:
    """Test whether two return series have significantly different Sharpe ratios.

    Answers "is my ΔSharpe vs buy-and-hold real, or sampling noise?" — the question a
    single Sharpe number cannot.

    Parameters
    ----------
    method:
        ``"jk_memmel"`` (default) — the Jobson-Korkie statistic with the Memmel (2003)
        correction, valid under iid returns:

        ``z = (s_a - s_b) / sqrt( (1/n) * [ 2(1-rho) + 0.5*(s_a^2 + s_b^2 - 2 s_a s_b rho^2) ] )``

        ``"bootstrap"`` — a circular-block bootstrap SE of the Sharpe difference (a
        Ledoit-Wolf-style HAC-robust variant) for autocorrelated returns.
    alpha:
        Two-sided significance level; ``passed`` reflects whether the difference is
        significant at ``alpha``.

    Returns
    -------
    CheckResult
        ``severity="info"`` (a Sharpe gap being significant is neither inherently good nor
        bad). ``statistic`` is the z-score; ``passed`` is ``p_value < alpha``. Sharpes and
        the difference are annualised in ``details`` via the clock.
    """
    a, b = _align(returns_a, returns_b)
    n = a.size
    if n < 3:
        raise ValueError(f"need at least 3 aligned observations, got {n}")

    s_a = _per_period_sharpe(a)
    s_b = _per_period_sharpe(b)
    diff = s_a - s_b

    if method == "jk_memmel":
        rho = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
        theta = (2.0 * (1.0 - rho) + 0.5 * (s_a**2 + s_b**2 - 2.0 * s_a * s_b * rho**2)) / n
        se = float(np.sqrt(theta)) if theta > 0 else float("nan")
    elif method == "bootstrap":
        bl = block_length if block_length is not None else max(1, int(round(n ** (1.0 / 3.0))))
        rng = np.random.default_rng(seed)
        diffs = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = _circular_block_indices(n, bl, rng)
            diffs[i] = _per_period_sharpe(a[idx]) - _per_period_sharpe(b[idx])
        se = float(np.std(diffs, ddof=1))
    else:
        raise ValueError(f"unknown method {method!r}; use 'jk_memmel' or 'bootstrap'")

    if not np.isfinite(se) or se <= 0:
        z = float("nan")
        p_value = float("nan")
        passed = False
    else:
        z = diff / se
        p_value = float(2.0 * norm.sf(abs(z)))
        passed = p_value < alpha

    sf = clock.sharpe_factor()
    details = (
        f"Annualized Sharpe A={s_a * sf:.3g}, B={s_b * sf:.3g}, "
        f"Δ={diff * sf:.3g} (per-period Δ={diff:.4g}); {method} z={z:.3g}, "
        f"p={p_value:.3g}. "
        + (
            f"The Sharpe difference is significant at alpha={alpha}."
            if passed
            else f"The Sharpe difference is NOT significant at alpha={alpha} — "
            "consistent with sampling noise."
        )
    )
    return CheckResult(
        name=name,
        passed=passed,
        severity="info",
        statistic=z if np.isfinite(z) else None,
        threshold=float(norm.isf(alpha / 2.0)),
        details=details,
    )
