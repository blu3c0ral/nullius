"""Deliberately leaky reference implementations.

The library's testing philosophy, stated verbatim: **"a leak detector you have never
seen fire is itself untested."** Every probe ships with a control that must fire, used
both in CI (a probe with no firing test fails review) and exported so users can
sanity-check their own harnesses against a known-bad baseline.

- :func:`leaky_global_quantile_gate` — the canonical lookahead bug: a threshold set from
  the *full-history* quantile and applied to every timestamp. :func:`check_prefix_invariance`
  must flag it.
- :func:`clean_expanding_quantile_gate` — the fixed version: an expanding quantile using
  only past-or-present data. It must pass.
- :func:`leaky_unpurged_splits` — walk-forward folds with train adjacent to test and no
  purge gap. :func:`check_split_purging` must flag them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .splits import Split


def leaky_global_quantile_gate(
    data: pd.DataFrame,
    *,
    q: float = 0.90,
    time_col: str = "time",
    pred_col: str = "predicted",
) -> pd.DataFrame:
    """LEAKY pipeline: gate on the full-sample quantile of ``pred_col``.

    The threshold is the ``q``-quantile computed over *every* row in the input and then
    applied to each timestamp — so truncating the input changes the threshold, and with
    it decisions dated before the truncation point. This is the lookahead bug that
    :func:`check_prefix_invariance` exists to catch.
    """
    out = data[[c for c in (time_col, "asset") if c in data.columns]].copy()
    threshold = float(np.quantile(data[pred_col].to_numpy(dtype=float), q))
    out["threshold"] = threshold
    out["signal"] = (data[pred_col].to_numpy(dtype=float) >= threshold).astype(int)
    return out


def clean_expanding_quantile_gate(
    data: pd.DataFrame,
    *,
    q: float = 0.90,
    time_col: str = "time",
    pred_col: str = "predicted",
) -> pd.DataFrame:
    """CLEAN pipeline: gate on an expanding quantile using only data up to each timestamp.

    For a row at time ``t`` the threshold is the ``q``-quantile over all rows with time
    ``<= t``. Because that window never includes future data, the decision for any row
    dated ``<= T`` is unchanged when the input is truncated at ``T`` — so this passes
    :func:`check_prefix_invariance`.
    """
    df = data.copy()
    times = pd.to_datetime(df[time_col])
    preds = df[pred_col].to_numpy(dtype=float)
    uniq = np.sort(pd.unique(times))
    thr_by_time = {}
    for t in uniq:
        mask = (times <= t).to_numpy()
        thr_by_time[t] = float(np.quantile(preds[mask], q))
    threshold = times.map(thr_by_time).to_numpy(dtype=float)

    out = df[[c for c in (time_col, "asset") if c in df.columns]].copy()
    out["threshold"] = threshold
    out["signal"] = (preds >= threshold).astype(int)
    return out


def leaky_unpurged_splits(
    times,
    *,
    train_span: int,
    test_span: int,
    step: int | None = None,
) -> list[Split]:
    """LEAKY splitter: walk-forward folds with train placed directly against test.

    No purge gap, so a forward-looking label at the last training bar overlaps the test
    window. :func:`check_split_purging` at any ``label_horizon >= 1`` must flag every
    fold.
    """
    n = int(len(times))
    step = test_span if step is None else int(step)
    splits: list[Split] = []
    test_start = train_span
    while test_start + test_span <= n:
        train_lo = max(0, test_start - train_span)
        splits.append(
            Split(
                train_idx=np.arange(train_lo, test_start, dtype=np.int64),
                test_idx=np.arange(test_start, test_start + test_span, dtype=np.int64),
            )
        )
        test_start += step
    return splits
