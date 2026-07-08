"""Placebo generators — nulls that a "real" strategy must beat.

Two ported generators (parity-eligible against the source repo) plus one new
trade-schedule shuffler:

- :func:`random_uniform_placebo` — predictions replaced by uniform noise in each asset's
  observed ``[min, max]``. Weak control: does not share the original autocorrelation.
- :func:`ar1_matched_placebo` — AR(1) noise matching each asset's lag-1 autocorrelation,
  mean and variance. Stronger: a strategy that only worked by riding autocorrelation
  looks the same as this placebo. Beating it rejects "no skill", **not** "no better than
  a skilled persistence forecaster" — a low bar; benchmark against a lag-1 baseline too.
- :func:`shuffled_entries_placebo` — keeps each asset's trade count and holding-period
  lengths (in bars) but randomly, non-overlappingly repositions entries on the bar grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PREDICTED_COL = "predicted"
ASSET_COL = "asset"


def _check_cols(predictions: pd.DataFrame, asset_col: str, pred_col: str) -> None:
    for col in (asset_col, pred_col):
        if col not in predictions.columns:
            raise ValueError(
                f"column {col!r} not found; available columns: {list(predictions.columns)}"
            )


def random_uniform_placebo(
    predictions: pd.DataFrame,
    *,
    seed: int = 42,
    pred_col: str = PREDICTED_COL,
    asset_col: str = ASSET_COL,
) -> pd.DataFrame:
    """Replace ``pred_col`` with uniform noise in each asset's observed ``[min, max]``.

    Other columns are preserved verbatim. Breaks the prediction/outcome relationship
    while staying inside the original prediction range, so it detects a strategy whose
    edge is an artefact of the raw prediction scale rather than its ordering.
    """
    _check_cols(predictions, asset_col, pred_col)
    rng = np.random.default_rng(seed)
    out = predictions.copy()
    for _, sub in predictions.groupby(asset_col):
        lo = float(sub[pred_col].min())
        hi = float(sub[pred_col].max())
        out.loc[sub.index, pred_col] = rng.uniform(lo, hi, size=len(sub))
    return out


def _ar1_lag1(series: pd.Series) -> float:
    s = series.dropna().astype(float)
    if len(s) < 2:
        return 0.0
    x = s.values[:-1]
    y = s.values[1:]
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def ar1_matched_placebo(
    predictions: pd.DataFrame,
    *,
    seed: int = 42,
    pred_col: str = PREDICTED_COL,
    asset_col: str = ASSET_COL,
) -> pd.DataFrame:
    """Replace ``pred_col`` with AR(1) noise matching each asset's lag-1 autocorrelation,
    mean and unconditional variance.

    Construction: ``x_t = mu + phi * (x_{t-1} - mu) + eps_t`` with ``phi`` = lag-1
    autocorrelation of the original series and ``eps_t ~ N(0, sigma^2 * (1 - phi^2))`` so
    the unconditional variance matches the original.

    Preserves lag-1 autocorrelation, mean and variance per asset; does **not** preserve
    distribution shape (skew/kurtosis), higher-order autocorrelation, or cross-asset
    correlation.
    """
    _check_cols(predictions, asset_col, pred_col)
    rng = np.random.default_rng(seed)
    out = predictions.copy()
    for _, sub in predictions.groupby(asset_col):
        orig = sub[pred_col].astype(float)
        mu = float(orig.mean())
        sigma = float(orig.std(ddof=1)) if len(orig) > 1 else 0.0
        phi = _ar1_lag1(orig)
        phi = max(min(phi, 0.999), -0.999)
        n = len(sub)
        if sigma == 0.0 or n == 0:
            out.loc[sub.index, pred_col] = mu
            continue
        eps_std = sigma * np.sqrt(max(1.0 - phi * phi, 1e-12))
        eps = rng.normal(0.0, eps_std, size=n)
        x = np.empty(n, dtype=float)
        x[0] = mu + sigma * rng.normal()
        for i in range(1, n):
            x[i] = mu + phi * (x[i - 1] - mu) + eps[i]
        out.loc[sub.index, pred_col] = x
    return out


def _random_composition(total: int, parts: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random composition of ``total`` into ``parts`` non-negative integers."""
    if parts == 1:
        return np.array([total], dtype=np.int64)
    if total == 0:
        return np.zeros(parts, dtype=np.int64)
    dividers = np.sort(rng.choice(total + parts - 1, parts - 1, replace=False))
    boundaries = np.concatenate(([-1], dividers, [total + parts - 1]))
    return (np.diff(boundaries) - 1).astype(np.int64)


def _bars_in_span(grid_pos: np.ndarray, entry: pd.Timestamp, exit_: pd.Timestamp) -> int:
    lo = int(np.searchsorted(grid_pos, entry.value, side="left"))
    hi = int(np.searchsorted(grid_pos, exit_.value, side="right"))
    return max(hi - lo, 0)


def shuffled_entries_placebo(
    trades: pd.DataFrame,
    times,
    *,
    seed: int = 42,
    asset_col: str = ASSET_COL,
    entry_col: str = "entry_time",
    exit_col: str = "exit_time",
) -> pd.DataFrame:
    """Randomly reposition each asset's trades on the bar grid, keeping count and lengths.

    For each asset the number of trades and each trade's holding length in **bars** are
    preserved, but entries are placed at uniformly random, non-overlapping positions on
    the ``times`` grid. This is a null for trade *timing*: it holds market exposure fixed
    and asks whether the edge comes from *when* the strategy trades or merely from being
    invested that much.

    Non-time columns (including ``net_return``) are carried through unchanged, so returns
    should be **recomputed from prices** on the placebo schedule — a shuffled entry no
    longer corresponds to its original realised return.

    Parameters
    ----------
    trades:
        Trades frame with ``asset_col``, ``entry_col``, ``exit_col``.
    times:
        The bar grid (any datetime-like sequence). Trades are repositioned onto its
        sorted unique timestamps.
    """
    for col in (asset_col, entry_col, exit_col):
        if col not in trades.columns:
            raise ValueError(f"column {col!r} not found in trades")

    grid = np.sort(pd.unique(pd.to_datetime(pd.Series(times))))
    grid_pos = grid.astype("datetime64[ns]").astype("int64")
    n_bars = grid.size
    if n_bars == 0:
        raise ValueError("times grid is empty")

    rng = np.random.default_rng(seed)
    df = trades.copy()
    df[entry_col] = pd.to_datetime(df[entry_col])
    df[exit_col] = pd.to_datetime(df[exit_col])

    out_rows: list[pd.Series] = []
    for _, sub in df.groupby(asset_col):
        sub = sub.sort_values(entry_col)
        lengths = np.array(
            [_bars_in_span(grid_pos, r[entry_col], r[exit_col]) for _, r in sub.iterrows()],
            dtype=np.int64,
        )
        lengths = np.maximum(lengths, 1)
        k = len(lengths)
        total_len = int(lengths.sum())
        free = n_bars - total_len
        if free < 0:
            raise ValueError(
                f"asset has {k} trades occupying {total_len} bars but the grid only has "
                f"{n_bars}; cannot place non-overlapping placebo entries"
            )
        order = rng.permutation(k)
        gaps = _random_composition(free, k + 1, rng)
        pos = int(gaps[0])
        # Placement is keyed by shuffled order; write results back to original rows.
        placed_start = np.empty(k, dtype=np.int64)
        placed_end = np.empty(k, dtype=np.int64)
        for j, oi in enumerate(order):
            start = pos
            end = start + int(lengths[oi]) - 1
            placed_start[oi] = start
            placed_end[oi] = end
            pos = end + 1 + int(gaps[j + 1])
        for row_i, (_, row) in enumerate(sub.iterrows()):
            new = row.copy()
            new[entry_col] = grid[placed_start[row_i]]
            new[exit_col] = grid[placed_end[row_i]]
            out_rows.append(new)

    result = pd.DataFrame(out_rows).reset_index(drop=True)
    return result[list(trades.columns)]
