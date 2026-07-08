"""Compare a real metric against a placebo distribution."""

from __future__ import annotations

import numpy as np

from ..result import CheckResult


def compare_against_placebos(
    real_metric: float,
    placebo_metrics,
    *,
    higher_is_better: bool = True,
    quantile_threshold: float = 0.95,
    name: str = "placebo_comparison",
) -> CheckResult:
    """Locate a real strategy's metric within its placebo distribution.

    Detects the failure mode where a headline number (Sharpe, Calmar, IC, ...) is no
    better than what a skill-free placebo produces — i.e. the "edge" is luck or an
    artefact the placebo also has. Reports the z-score of the real metric against the
    placebos, its rank, and the fraction of placebos it beats; passes when it beats at
    least ``quantile_threshold`` of them.

    Parameters
    ----------
    real_metric:
        The real strategy's metric.
    placebo_metrics:
        The same metric computed on N placebo runs (e.g. seeded AR(1) placebos).
    higher_is_better:
        Whether a larger metric is better (Sharpe/Calmar) or smaller (a loss).
    quantile_threshold:
        Fraction of placebos the real metric must beat to pass (default 0.95).

    Notes
    -----
    Beating a placebo is a **necessary, not sufficient** condition. AR(1)-matched noise
    is a weak null; passing rejects "no skill", not "no better than a naive persistent
    forecaster". Hence ``severity="warning"``: a failure is a real red flag, a pass is
    only a floor cleared.
    """
    arr = np.asarray(list(placebo_metrics), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 2:
        raise ValueError(f"need at least 2 finite placebo metrics to form a distribution, got {n}")
    if not np.isfinite(real_metric):
        raise ValueError(f"real_metric must be finite, got {real_metric}")

    beaten = arr < real_metric if higher_is_better else arr > real_metric
    frac_beaten = float(np.mean(beaten))
    rank = int(np.sum(beaten)) + 1  # 1-based rank of real among (placebos + real)

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std > 0:
        z = (real_metric - mean) / std
        z = z if higher_is_better else -z
    else:
        z = float("inf") if beaten.all() else (0.0 if frac_beaten == 0 else float("nan"))

    passed = frac_beaten >= quantile_threshold
    details = (
        f"Real metric {real_metric:.6g} beats {frac_beaten:.0%} of {n} placebo(s) "
        f"(rank {rank} of {n + 1}, z={z:.3g}). "
        + (
            "Clears the placebo floor — but note this only rejects 'no skill'; "
            "benchmark against a skilled naive baseline (e.g. lag-1) too."
            if passed
            else "Does NOT clear the placebo floor: the metric is indistinguishable "
            "from a skill-free null. The apparent edge is likely luck or an artefact "
            "the placebo shares."
        )
    )
    return CheckResult(
        name=name,
        passed=passed,
        severity="warning",
        statistic=frac_beaten,
        threshold=quantile_threshold,
        details=details,
    )
