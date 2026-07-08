"""Purged / embargoed walk-forward splits, and a checker for externally built folds.

The failure mode: a label computed at bar ``i`` that looks ``label_horizon`` bars into
the future (a forward return, a triple-barrier outcome) leaks into a test set that
starts within ``label_horizon`` bars of ``i``. Naive walk-forward that puts train
immediately adjacent to test is contaminated by exactly this overlap.

Per the timeframe-generic rule, ``train_span``, ``label_horizon`` and ``embargo`` are
all expressed in **bars (observation counts), never calendar days**.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..result import CheckResult


@dataclass(frozen=True)
class Split:
    """One walk-forward fold as positional integer indices into a sorted time axis."""

    train_idx: np.ndarray
    test_idx: np.ndarray


def purged_splits(
    times,
    *,
    train_span: int,
    test_span: int,
    label_horizon: int,
    embargo: int = 0,
    step: int | None = None,
    expanding: bool = False,
) -> list[Split]:
    """Build walk-forward splits that purge the ``label_horizon``+``embargo`` bars
    separating each train window from its test window.

    Parameters
    ----------
    times:
        Any sequence whose *length* and ordering define the observation axis; only
        ``len(times)`` is used (indices are positional). Pass the sorted time key.
    train_span:
        Number of training bars per fold (rolling); ignored when ``expanding=True``,
        where training starts at bar 0.
    test_span:
        Number of test bars per fold.
    label_horizon:
        Forward span (in bars) of the label. The last ``label_horizon`` train bars
        before each test window are dropped so no training label peeks into the test.
    embargo:
        Additional bars dropped after the purge gap (López de Prado embargo).
    step:
        Bars to advance the test window between folds; defaults to ``test_span``
        (non-overlapping test blocks).
    expanding:
        If ``True``, each fold trains on all bars from 0 up to the purge gap; if
        ``False`` (default), on a rolling window of ``train_span`` bars.

    Returns
    -------
    list of :class:`Split`
        Folds in chronological order. Guaranteed purged: every returned fold passes
        :func:`check_split_purging` at the same ``label_horizon``.
    """
    for name, val in (
        ("train_span", train_span),
        ("test_span", test_span),
    ):
        if val <= 0:
            raise ValueError(f"{name} must be positive, got {val}")
    if label_horizon < 0 or embargo < 0:
        raise ValueError("label_horizon and embargo must be non-negative")
    n = int(len(times))
    step = test_span if step is None else int(step)
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")

    splits: list[Split] = []
    test_start = train_span + label_horizon + embargo
    while test_start + test_span <= n:
        train_end = test_start - label_horizon - embargo  # exclusive upper bound
        train_lo = 0 if expanding else max(0, train_end - train_span)
        train_idx = np.arange(train_lo, train_end, dtype=np.int64)
        test_idx = np.arange(test_start, test_start + test_span, dtype=np.int64)
        if train_idx.size > 0:
            splits.append(Split(train_idx=train_idx, test_idx=test_idx))
        test_start += step
    return splits


def _as_split(item) -> Split:
    if isinstance(item, Split):
        return item
    train_idx, test_idx = item
    return Split(
        train_idx=np.asarray(train_idx, dtype=np.int64),
        test_idx=np.asarray(test_idx, dtype=np.int64),
    )


def check_split_purging(
    splits: Iterable[Split] | Iterable[tuple],
    *,
    label_horizon: int,
) -> CheckResult:
    """Flag walk-forward folds whose train/test boundary is not purged.

    Detects the exact leak that unpurged walk-forward introduces: a training row whose
    ``label_horizon``-bar forward label overlaps the test window. For each fold a leak
    is present when ``max(train_idx) >= min(test_idx) - label_horizon`` — i.e. the last
    training bar sits within the label's forward reach of the first test bar. Accepts
    :class:`Split` objects or raw ``(train_idx, test_idx)`` index pairs, so it audits
    folds produced by any splitter (sklearn, a hand-rolled loop, ...).

    Why it matters: unpurged folds inflate out-of-sample skill because the model was
    trained on labels that already contain the test period's outcomes.
    """
    if label_horizon < 0:
        raise ValueError(f"label_horizon must be non-negative, got {label_horizon}")

    rows = []
    n_folds = 0
    for i, raw in enumerate(splits):
        fold = _as_split(raw)
        n_folds += 1
        if fold.train_idx.size == 0 or fold.test_idx.size == 0:
            continue
        max_train = int(fold.train_idx.max())
        min_test = int(fold.test_idx.min())
        gap = min_test - max_train - 1  # purged bars strictly between train and test
        if max_train >= min_test - label_horizon:
            rows.append(
                {
                    "fold": i,
                    "max_train_idx": max_train,
                    "min_test_idx": min_test,
                    "gap_bars": gap,
                    "required_gap": label_horizon,
                }
            )

    n_leaky = len(rows)
    passed = n_leaky == 0
    evidence = None if passed else pd.DataFrame(rows)
    details = (
        f"All {n_folds} fold(s) leave at least {label_horizon} purged bar(s) between "
        "train and test."
        if passed
        else f"{n_leaky} of {n_folds} fold(s) place training bars within "
        f"{label_horizon} bar(s) of the test window; their forward-looking labels "
        "overlap the test period and inflate out-of-sample skill."
    )
    return CheckResult(
        name="split_purging",
        passed=passed,
        severity="fatal",
        statistic=float(n_leaky),
        threshold=0.0,
        details=details,
        evidence=evidence,
    )
