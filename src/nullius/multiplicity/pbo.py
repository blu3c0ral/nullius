"""Probability of Backtest Overfitting via CSCV — Bailey, Borwein, López de Prado, Zhu.

Combinatorially-Symmetric Cross-Validation asks: across many train/test partitions, how
often does the configuration that looked best in-sample land in the bottom half
out-of-sample? A high rate means the selection procedure is picking overfit noise rather
than signal.

``probability_of_backtest_overfitting`` is ported verbatim from the source repo (golden
parity). ``pbo`` is a :class:`CheckResult` wrapper with the standard pass/warning/fatal
bands.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from ..result import CheckResult


def _sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else 0.0


def probability_of_backtest_overfitting(
    returns_matrix,
    *,
    n_splits: int = 16,
    max_combos: int = 3000,
    seed: int = 42,
) -> dict:
    """PBO via CSCV.

    Parameters
    ----------
    returns_matrix:
        Array-like or DataFrame, shape (T, N) — per-period returns for N candidate
        CONFIGURATIONS (strategy columns only; never a benchmark column or a date index).
    n_splits:
        Number of contiguous row blocks S (must be even; T >= 2*S).
    max_combos:
        Cap on the C(S, S/2) IS/OOS combinations evaluated (subsampled, seeded, if
        exceeded).
    seed:
        RNG seed for the subsample.

    Returns
    -------
    dict
        Keys: ``pbo`` (fraction of combinations whose IS-best config lands in the OOS
        bottom half, i.e. logit ``lambda <= 0``), ``lambda_dist`` (list), ``median_logit``,
        ``n_combos``, ``n_configs``, ``n_splits``. Rule of thumb: < 0.20 ok, < 0.50
        warning, >= 0.50 fail.
    """
    M = np.asarray(
        returns_matrix.values if isinstance(returns_matrix, pd.DataFrame) else returns_matrix,
        dtype=float,
    )
    if M.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (T x N)")
    T, N = M.shape
    if N < 2:
        raise ValueError(f"need >= 2 configurations for CSCV, got {N}")
    if n_splits % 2 != 0:
        raise ValueError(f"n_splits must be even, got {n_splits}")
    if 2 * n_splits > T:
        raise ValueError(f"need T >= 2*n_splits ({2 * n_splits}) rows for CSCV, got T={T}")

    blocks = np.array_split(np.arange(T), n_splits)
    all_combos = list(combinations(range(n_splits), n_splits // 2))
    rng = np.random.default_rng(seed)
    if len(all_combos) > max_combos:
        pick = rng.choice(len(all_combos), size=max_combos, replace=False)
        combos = [all_combos[i] for i in pick]
    else:
        combos = all_combos

    lambdas = []
    for is_blocks in combos:
        is_set = set(is_blocks)
        is_rows = np.concatenate([blocks[b] for b in is_blocks])
        oos_rows = np.concatenate([blocks[b] for b in range(n_splits) if b not in is_set])
        is_sr = np.array([_sharpe(M[is_rows, j]) for j in range(N)])
        oos_sr = np.array([_sharpe(M[oos_rows, j]) for j in range(N)])
        n_star = int(np.argmax(is_sr))
        # relative OOS rank of the IS-best config, in (0, 1)
        order = oos_sr.argsort()  # ascending
        ranks = np.empty(N, dtype=float)
        ranks[order] = np.arange(1, N + 1)
        omega = ranks[n_star] / (N + 1.0)
        lambdas.append(float(np.log(omega / (1.0 - omega))))

    lam = np.asarray(lambdas, dtype=float)
    return {
        "pbo": float(np.mean(lam <= 0.0)),
        "lambda_dist": [float(x) for x in lam],
        "median_logit": float(np.median(lam)),
        "n_combos": int(len(combos)),
        "n_configs": int(N),
        "n_splits": int(n_splits),
    }


def pbo(
    returns_matrix,
    *,
    n_splits: int = 16,
    max_combos: int = 3000,
    seed: int = 42,
) -> CheckResult:
    """PBO as a :class:`CheckResult` with the standard bands.

    Detects an overfit configuration search: passes when PBO < 0.20, warns in
    ``[0.20, 0.50)``, and is fatal at ``>= 0.50`` (the IS-best config is no better than a
    coin flip out-of-sample). The logit ``lambda`` distribution is attached as evidence.
    """
    d = probability_of_backtest_overfitting(
        returns_matrix, n_splits=n_splits, max_combos=max_combos, seed=seed
    )
    p = d["pbo"]
    passed = p < 0.20
    severity = "fatal" if p >= 0.50 else "warning"
    details = f"PBO={p:.3f} over {d['n_combos']} CSCV split(s) of {d['n_configs']} configs. " + (
        "The in-sample-best configuration reliably stays strong out-of-sample."
        if passed
        else "The in-sample-best configuration frequently lands in the "
        "out-of-sample bottom half — the search is selecting overfit noise."
    )
    return CheckResult(
        name="pbo",
        passed=passed,
        severity=severity,
        statistic=p,
        threshold=0.20,
        details=details,
        evidence=pd.DataFrame({"logit_lambda": d["lambda_dist"]}),
    )
