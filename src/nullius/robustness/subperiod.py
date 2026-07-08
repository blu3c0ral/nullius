"""Regime / subperiod stratification — is the headline number distributed or concentrated?

A Sharpe that lives entirely in one regime (a single crash, a single recovery) is fragile.
These helpers split the sample into named periods and report per-period IC and per-period
risk-adjusted metrics, so concentration is visible rather than hidden inside a full-sample
average.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..clock import DAILY, Clock
from ..metrics.equity import equity_metrics

#: Labeled EXAMPLE regime dict for US equities. Illustrative only — supply your own.
US_EQUITY_REGIMES: dict[str, tuple[str, str]] = {
    "pre_covid": ("2018-01-01", "2020-02-19"),
    "covid_crash": ("2020-02-20", "2020-03-23"),
    "covid_recovery": ("2020-03-24", "2021-12-31"),
    "bear_2022": ("2022-01-01", "2022-10-12"),
    "recovery_2023_plus": ("2022-10-13", "2025-12-31"),
}


def _slice(df: pd.DataFrame, start, end, time_col: str) -> pd.DataFrame:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    t = pd.to_datetime(df[time_col])
    return df.loc[(t >= s) & (t <= e)]


def subperiod_ic_stability(
    predictions: pd.DataFrame,
    periods: Mapping[str, tuple[str, str]],
    *,
    per_asset: bool = True,
    time_col: str = "time",
    asset_col: str = "asset",
    pred_col: str = "predicted",
    actual_col: str = "actual",
    min_obs: int = 20,
) -> pd.DataFrame:
    """Spearman IC within each named period.

    ``periods`` maps ``name -> (start, end)``. With ``per_asset=True`` returns one row per
    (period, asset); otherwise one pooled row per period. Per-asset cells with fewer than
    ``min_obs`` observations are skipped (Spearman is too noisy). Pooled IC across assets
    is statistically dubious (violates iid) and only offered for completeness.
    """
    rows = []
    for name, (start, end) in periods.items():
        period_df = _slice(predictions, start, end, time_col)
        if period_df.empty:
            continue
        if per_asset:
            for tk, sub in period_df.groupby(asset_col):
                mask = sub[pred_col].notna() & sub[actual_col].notna()
                if mask.sum() < min_obs:
                    continue
                r, _ = spearmanr(sub.loc[mask, pred_col], sub.loc[mask, actual_col])
                rows.append(
                    {
                        "period": name,
                        "start": start,
                        "end": end,
                        "asset": tk,
                        "ic": float(r) if not np.isnan(r) else np.nan,
                        "n_obs": int(mask.sum()),
                    }
                )
        else:
            mask = period_df[pred_col].notna() & period_df[actual_col].notna()
            if mask.sum() < 2:
                continue
            r, _ = spearmanr(period_df.loc[mask, pred_col], period_df.loc[mask, actual_col])
            rows.append(
                {
                    "period": name,
                    "start": start,
                    "end": end,
                    "asset": "ALL",
                    "ic": float(r) if not np.isnan(r) else np.nan,
                    "n_obs": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def regime_stratified_metrics(
    portfolio_returns: pd.Series,
    periods: Mapping[str, tuple[str, str]],
    *,
    clock: Clock = DAILY,
) -> pd.DataFrame:
    """Calmar / Sharpe / MDD by named regime, from a time-indexed return series.

    Each regime's metrics are computed on that period's slice **in isolation**: the equity
    curve is rebuilt from 1.0 at the slice start, so a drawdown that began before the period
    is measured only from the in-slice peak, not carried across the boundary. This answers
    "how did the strategy do *within* each regime", not "where did each drawdown start". To
    measure the latter, compute the drawdown on the full-sample curve and slice that series.
    """
    r = portfolio_returns.dropna().astype(float)
    r.index = pd.to_datetime(r.index)
    rows = []
    for name, (start, end) in periods.items():
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        sub = r.loc[(r.index >= s) & (r.index <= e)]
        if sub.empty:
            continue
        m = equity_metrics(sub, clock=clock)
        m.update({"period": name, "start": start, "end": end})
        rows.append(m)
    return pd.DataFrame(rows)
