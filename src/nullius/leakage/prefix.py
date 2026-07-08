"""Prefix-invariance — the flagship leakage probe.

Principle (the general form of a real lookahead bug that survived four weeks of a
"careful" research process): **any output timestamped at or before T must be identical
whether the pipeline saw data up to T or all the way to today.** A global statistic
computed once over the whole sample — a full-history quantile threshold, a full-sample
mean/std normaliser, a post-hoc calibration — and then applied inside a per-timestamp
decision violates this, because truncating the input at T changes that global statistic
and therefore changes decisions dated ``<= T`` that should have been frozen.

The probe needs zero knowledge of the pipeline internals: it re-runs the pipeline on a
truncated copy of the input and compares the two outputs on the rows they share.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from ..result import CheckResult


def _default_cut_times(times: pd.Series) -> np.ndarray:
    """Five timestamps evenly spaced across the middle 60% of the sample."""
    uniq = np.sort(pd.unique(times.dropna()))
    if uniq.size < 2:
        return uniq
    positions = np.linspace(0.2, 0.8, 5) * (uniq.size - 1)
    idx = np.unique(np.round(positions).astype(int))
    return uniq[idx]


def _mismatch_rows(
    full_le: pd.DataFrame,
    trunc_le: pd.DataFrame,
    keys: list[str],
    atol: float,
    cut,
) -> pd.DataFrame:
    """Rows that differ between the two truncations, restricted to ``time <= cut``."""
    f = full_le.set_index(keys).sort_index()
    t = trunc_le.set_index(keys).sort_index()

    if f.index.has_duplicates or t.index.has_duplicates:
        return pd.DataFrame(
            [
                {
                    "cut_time": cut,
                    "column": "(index)",
                    "issue": "pipeline output is not uniquely keyed by "
                    f"{keys}; cannot align for comparison",
                    "full_value": np.nan,
                    "truncated_value": np.nan,
                }
            ]
        )

    rows: list[dict] = []
    only_full = f.index.difference(t.index)
    only_trunc = t.index.difference(f.index)
    for idx in list(only_full):
        rows.append(_membership_row(keys, idx, cut, present_in="full"))
    for idx in list(only_trunc):
        rows.append(_membership_row(keys, idx, cut, present_in="truncated"))

    common = f.index.intersection(t.index)
    shared_cols = [c for c in f.columns if c in t.columns]
    for col in shared_cols:
        fv = f.loc[common, col]
        tv = t.loc[common, col]
        if pdt.is_numeric_dtype(fv) and pdt.is_numeric_dtype(tv):
            fa = fv.to_numpy(dtype=float)
            ta = tv.to_numpy(dtype=float)
            both_nan = np.isnan(fa) & np.isnan(ta)
            differ = (~both_nan) & ~(np.abs(fa - ta) <= atol)
        else:
            fna, tna = fv.isna().to_numpy(), tv.isna().to_numpy()
            differ = (fv.to_numpy() != tv.to_numpy()) & ~(fna & tna)
        for pos in np.flatnonzero(differ):
            key_val = common[pos]
            rows.append(
                {
                    **_key_dict(keys, key_val),
                    "cut_time": cut,
                    "column": col,
                    "full_value": fv.iloc[pos],
                    "truncated_value": tv.iloc[pos],
                    "issue": "value changed when input was truncated",
                }
            )
    return pd.DataFrame(rows)


def _key_dict(keys: list[str], key_val) -> dict:
    if len(keys) == 1:
        return {keys[0]: key_val}
    return dict(zip(keys, key_val, strict=True))


def _membership_row(keys, idx, cut, present_in) -> dict:
    other = "truncated" if present_in == "full" else "full"
    return {
        **_key_dict(keys, idx),
        "cut_time": cut,
        "column": "(row)",
        "full_value": np.nan,
        "truncated_value": np.nan,
        "issue": f"row present in {present_in} output but absent in {other} output",
    }


def check_prefix_invariance(
    pipeline: Callable[[pd.DataFrame], pd.DataFrame],
    data: pd.DataFrame,
    cut_times: Sequence | None = None,
    *,
    time_col: str = "time",
    atol: float = 1e-9,
) -> CheckResult:
    """Verify that outputs dated ``<= T`` do not change when the input is truncated at T.

    Parameters
    ----------
    pipeline:
        Maps an input frame to a timestamped output frame (signals, thresholds,
        trades, or predictions). Must return a frame containing ``time_col``; if the
        output also has an ``asset`` column it is used as a secondary alignment key.
    data:
        The full input frame; must contain ``time_col``.
    cut_times:
        Timestamps at which to truncate. Defaults to five evenly spaced across the
        middle 60% of the sample.
    atol:
        Absolute tolerance for numeric equality.

    Returns
    -------
    CheckResult
        ``severity="fatal"``. ``statistic`` is the total number of mismatched rows
        across all cut times; ``evidence`` lists them. A mismatch means some decision
        dated ``<= T`` depended on data after ``T`` — a lookahead leak.

    Notes
    -----
    Walk-forward retraining legitimately changes outputs dated ``> T``; only rows dated
    ``<= T`` are compared, so honest walk-forward passes. This probe works at timestamp
    granularity: it validates "no future *bars*", not intra-*bar* fill leakage (deciding
    on ``close_t`` and filling the same bar). That intraday failure mode is covered by
    the ``nullius[intraday]`` execution-timing probe.
    """
    if time_col not in data.columns:
        raise ValueError(f"data must contain time_col {time_col!r}")

    times = pd.to_datetime(data[time_col])
    cuts = _default_cut_times(times) if cut_times is None else np.asarray(cut_times)

    full = pipeline(data)
    if time_col not in full.columns:
        raise ValueError(
            f"pipeline output must contain time_col {time_col!r}; got columns {list(full.columns)}"
        )
    full = full.copy()
    full[time_col] = pd.to_datetime(full[time_col])

    evidence_frames: list[pd.DataFrame] = []
    for cut in cuts:
        cut_ts = pd.Timestamp(cut)
        sub = data[pd.to_datetime(data[time_col]) <= cut_ts]
        if sub.empty:
            continue
        trunc = pipeline(sub).copy()
        if time_col not in trunc.columns:
            raise ValueError(
                f"pipeline output must contain time_col {time_col!r}; got columns "
                f"{list(trunc.columns)}"
            )
        trunc[time_col] = pd.to_datetime(trunc[time_col])

        keys = [time_col]
        if "asset" in full.columns and "asset" in trunc.columns:
            keys.append("asset")

        f_le = full[full[time_col] <= cut_ts]
        t_le = trunc[trunc[time_col] <= cut_ts]
        m = _mismatch_rows(f_le, t_le, keys, atol, cut_ts)
        if not m.empty:
            evidence_frames.append(m)

    n_mismatch = int(sum(len(m) for m in evidence_frames))
    passed = n_mismatch == 0
    evidence = pd.concat(evidence_frames, ignore_index=True) if evidence_frames else None
    details = (
        f"Outputs dated <= T were identical under truncation at all {len(cuts)} cut "
        "time(s): no lookahead detected at bar granularity."
        if passed
        else f"{n_mismatch} output row(s) dated <= T changed when the input was "
        "truncated. Some decision at or before T depended on data after T — a "
        "lookahead leak (e.g. a full-sample statistic used inside a per-timestamp "
        "decision). See evidence."
    )
    return CheckResult(
        name="prefix_invariance",
        passed=passed,
        severity="fatal",
        statistic=float(n_mismatch),
        threshold=0.0,
        details=details,
        evidence=evidence,
    )
