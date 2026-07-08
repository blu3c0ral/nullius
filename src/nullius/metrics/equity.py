"""Equity-curve construction and risk-adjusted metrics.

Ported from the source repo's ``profitability`` module (golden-parity tested), with
vol_spike specifics dropped and column names generalised to the ``time``/``asset``
contract. The load-bearing choice preserved from the source: the equity curve is
reconstructed from prices and entry/exit anchors, **not** from a per-trade ``net_return``
column — a per-trade-compound approximation understates true drawdowns.

Every annualisation goes through the :class:`~nullius.clock.Clock`, so the same code is
correct at any fixed-bar frequency; ``DAILY`` supplies the 252/√252 defaults.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..clock import DAILY, Clock


def build_equity_curve(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    universe: Iterable[str],
    *,
    cost_per_side_bps: float = 6.0,
    time_col: str = "time",
    asset_col: str = "asset",
    close_col: str = "close",
    open_col: str = "open",
    entry_col: str = "entry_time",
    exit_col: str = "exit_time",
    exit_price_col: str = "exit_price",
    cash_period_yield: pd.Series | None = None,
) -> pd.DataFrame:
    """Build a per-bar per-asset P&L frame and an equal-weight portfolio curve.

    Each asset contributes ``1/N`` to the portfolio return where ``N = len(universe)``;
    an out-of-position asset contributes its cash yield (0 by default). Entry-bar P&L is
    ``close/open - 1`` less one-way cost; middle bars are close-to-close; the exit bar is
    ``exit_price/prev_close - 1`` less one-way cost. Overlapping trades for one asset
    raise, since the writer would silently overwrite them.

    Returns
    -------
    DataFrame
        Long form with columns ``time``, ``asset``, ``per_asset_ret`` (already divided by
        N, the portfolio contribution), and ``portfolio_ret`` (sum across assets, repeated
        per row).
    """
    universe = list(universe)
    n_assets = len(universe)
    if n_assets == 0:
        raise ValueError("universe must be non-empty")
    weight = 1.0 / n_assets
    cost_one_way = cost_per_side_bps / 1e4

    pr = prices[[time_col, asset_col, close_col, open_col]].copy()
    pr[time_col] = pd.to_datetime(pr[time_col])
    pr = pr[pr[asset_col].isin(universe)]
    pr = pr.drop_duplicates(subset=[time_col, asset_col], keep="last")
    pr = pr.sort_values([asset_col, time_col])

    all_times = pd.DatetimeIndex(sorted(pr[time_col].unique()))

    if cash_period_yield is None:
        period_ret = pd.DataFrame(0.0, index=all_times, columns=universe, dtype=float)
    else:
        cy = pd.Series(cash_period_yield).astype(float)
        cy.index = pd.to_datetime(cy.index)
        cy = cy.reindex(all_times).fillna(0.0)
        period_ret = pd.DataFrame(
            np.repeat(cy.to_numpy()[:, None], n_assets, axis=1),
            index=all_times,
            columns=universe,
            dtype=float,
        )

    if trades is not None and not trades.empty:
        tr = trades.copy()
        tr[entry_col] = pd.to_datetime(tr[entry_col])
        tr[exit_col] = pd.to_datetime(tr[exit_col])

        # Overlap protection: the per-trade writer uses .loc[t, asset] = value and would
        # silently overwrite if two trades on one asset overlap.
        for asset, grp in tr.groupby(asset_col):
            ordered = grp.sort_values(entry_col).reset_index(drop=True)
            for i in range(1, len(ordered)):
                prev = ordered.iloc[i - 1]
                cur = ordered.iloc[i]
                if cur[entry_col] <= prev[exit_col]:
                    raise ValueError(
                        f"Overlapping trades detected for asset {asset}: "
                        f"{(prev[entry_col], prev[exit_col])} and "
                        f"{(cur[entry_col], cur[exit_col])} overlap. build_equity_curve "
                        "does not support concurrent positions per asset."
                    )

        close_lut = pr.set_index([asset_col, time_col])[close_col]
        open_lut = pr.set_index([asset_col, time_col])[open_col]

        for _, t in tr.iterrows():
            asset = t[asset_col]
            if asset not in universe:
                continue
            entry_time = t[entry_col]
            exit_time = t[exit_col]
            exit_price = float(t[exit_price_col])

            mask = (all_times >= entry_time) & (all_times <= exit_time)
            hold_times = all_times[mask]
            if len(hold_times) == 0:
                continue

            try:
                entry_open = float(open_lut.loc[(asset, hold_times[0])])
                entry_close = float(close_lut.loc[(asset, hold_times[0])])
            except KeyError:
                continue
            entry_ret = entry_close / entry_open - 1.0 - cost_one_way

            if len(hold_times) == 1:
                entry_ret = exit_price / entry_open - 1.0 - 2.0 * cost_one_way
                period_ret.loc[hold_times[0], asset] = entry_ret
                continue

            period_ret.loc[hold_times[0], asset] = entry_ret

            for d_prev, d in zip(hold_times[:-1], hold_times[1:-1], strict=False):
                try:
                    prev_close = float(close_lut.loc[(asset, d_prev)])
                    cur_close = float(close_lut.loc[(asset, d)])
                except KeyError:
                    continue
                period_ret.loc[d, asset] = cur_close / prev_close - 1.0

            exit_t = hold_times[-1]
            prev_t = hold_times[-2]
            try:
                prev_close = float(close_lut.loc[(asset, prev_t)])
            except KeyError:
                continue
            period_ret.loc[exit_t, asset] = exit_price / prev_close - 1.0 - cost_one_way

    contrib = period_ret * weight
    portfolio = contrib.sum(axis=1)

    long = (
        contrib.stack()
        .reset_index()
        .rename(columns={"level_0": time_col, "level_1": asset_col, 0: "per_asset_ret"})
    )
    if time_col not in long.columns:
        long = long.rename(columns={long.columns[0]: time_col})
    if asset_col not in long.columns:
        long = long.rename(columns={long.columns[1]: asset_col})
    long["portfolio_ret"] = long[time_col].map(portfolio)
    return long


def equity_metrics(
    portfolio_returns: pd.Series,
    *,
    clock: Clock = DAILY,
    rf_annual: float = 0.0,
    rf_period_series: pd.Series | None = None,
) -> dict:
    """Sharpe / Sortino / Calmar / MDD / CAGR / vol from a per-period return series.

    Parameters
    ----------
    portfolio_returns:
        Per-period returns indexed by time.
    clock:
        Supplies ``periods_per_year`` (annualisation) and the Sharpe √-time factor.
    rf_annual:
        Scalar annual risk-free rate. **Defaulted to 0 deliberately and prominently:** a
        strategy that parks idle capital in cash at 0% during a 5% rate era looks better
        than it is. Set this (or ``rf_period_series``) to the real cash rate.
    rf_period_series:
        Optional time-indexed per-period risk-free series; when supplied, excess returns
        use it instead of the scalar — the correct benchmark when the risk-free rate moves.
    """
    r = portfolio_returns.dropna().astype(float)
    ppy = clock.periods_per_year
    sqrt_ppy = clock.sharpe_factor()
    if r.empty:
        return {
            "cagr": 0.0,
            "vol_ann": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "mdd": 0.0,
            "calmar": 0.0,
            "total_return": 0.0,
            "n_periods": 0,
            "first_time": None,
            "last_time": None,
        }

    if rf_period_series is not None:
        rf = pd.Series(rf_period_series).astype(float)
        rf.index = pd.to_datetime(rf.index)
        rf = rf.reindex(r.index).fillna(0.0)
        excess = r - rf
    else:
        excess = r - rf_annual / ppy

    equity = (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)

    years = len(r) / ppy
    if equity.iloc[-1] <= 0:
        cagr = -1.0
    elif years > 0:
        cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    else:
        cagr = 0.0

    ex_sd = float(excess.std(ddof=1))
    vol_ann = float(ex_sd * sqrt_ppy)
    sharpe = float(excess.mean() / ex_sd * sqrt_ppy) if ex_sd > 0 else 0.0

    neg_part = np.minimum(excess, 0.0)
    downside_dev = float(np.sqrt(np.mean(neg_part**2)))
    sortino = float(excess.mean() / downside_dev * sqrt_ppy) if downside_dev > 0 else 0.0

    peak = equity.cummax()
    dd = equity / peak - 1.0
    mdd = float(dd.min())
    calmar = float(cagr / abs(mdd)) if abs(mdd) > 1e-6 else 0.0

    return {
        "cagr": cagr,
        "vol_ann": vol_ann,
        "sharpe": sharpe,
        "sortino": sortino,
        "mdd": mdd,
        "calmar": calmar,
        "total_return": total_return,
        "n_periods": int(len(r)),
        "first_time": r.index[0],
        "last_time": r.index[-1],
    }


def buy_and_hold_curve(
    prices: pd.DataFrame,
    universe: Iterable[str],
    *,
    time_col: str = "time",
    asset_col: str = "asset",
    close_col: str = "close",
    start=None,
    end=None,
    rebalanced: bool = True,
) -> pd.DataFrame:
    """Equal-weight passive benchmark curve for the same universe.

    ``rebalanced=True`` (default) resets weights to ``1/N`` every bar; ``rebalanced=False``
    is true buy-and-hold from the first bar where all assets have prices, letting weights
    drift.

    Returns
    -------
    DataFrame
        Columns ``time``, ``bh_ret``, ``bh_equity``.
    """
    universe = list(universe)
    pr = prices[[time_col, asset_col, close_col]].copy()
    pr[time_col] = pd.to_datetime(pr[time_col])
    pr = pr[pr[asset_col].isin(universe)]
    if start is not None:
        pr = pr[pr[time_col] >= pd.Timestamp(start)]
    if end is not None:
        pr = pr[pr[time_col] <= pd.Timestamp(end)]
    pr = pr.drop_duplicates(subset=[time_col, asset_col], keep="last")

    wide = pr.pivot(index=time_col, columns=asset_col, values=close_col).sort_index()

    if rebalanced:
        daily = wide.pct_change(fill_method=None)
        bh_ret = daily.mean(axis=1, skipna=True).fillna(0.0)
        bh_equity = (1.0 + bh_ret).cumprod()
    else:
        all_present = wide.dropna(how="any")
        if all_present.empty:
            return pd.DataFrame({time_col: [], "bh_ret": [], "bh_equity": []})
        start_time = all_present.index[0]
        anchored = wide.loc[start_time:].copy()
        n = anchored.shape[1]
        start_prices = anchored.iloc[0]
        positions = anchored.divide(start_prices, axis=1) * (1.0 / n)
        bh_equity = positions.sum(axis=1)
        bh_ret = bh_equity.pct_change().fillna(0.0)

    return pd.DataFrame(
        {time_col: bh_equity.index, "bh_ret": bh_ret.values, "bh_equity": bh_equity.values}
    )


def exposure_stats(
    trades: pd.DataFrame,
    universe: Iterable[str],
    *,
    clock: Clock = DAILY,
    asset_col: str = "asset",
    entry_col: str = "entry_time",
    exit_col: str = "exit_time",
    test_start=None,
    test_end=None,
) -> pd.DataFrame:
    """Per-asset trade count and market-exposure statistics.

    A strategy's CAGR is only interpretable next to how often it was actually in the
    market. Returns one row per asset with ``n_trades``, ``n_trades_per_year``,
    ``total_hold_periods`` and ``pct_periods_in_market`` (business-day counts, so this is a
    daily-preset-oriented helper).
    """
    universe = list(universe)
    ppy = clock.periods_per_year

    def empty_row(tk):
        return {
            "asset": tk,
            "n_trades": 0,
            "n_trades_per_year": 0.0,
            "total_hold_periods": 0,
            "pct_periods_in_market": 0.0,
        }

    if trades is None or trades.empty:
        return pd.DataFrame([empty_row(tk) for tk in universe])

    t = trades.copy()
    t[entry_col] = pd.to_datetime(t[entry_col])
    t[exit_col] = pd.to_datetime(t[exit_col])
    if test_start is not None:
        t = t[t[entry_col] >= pd.Timestamp(test_start)]
    if test_end is not None:
        t = t[t[exit_col] <= pd.Timestamp(test_end)]

    if test_start is None or test_end is None:
        if not t.empty:
            ts = pd.Timestamp(test_start) if test_start is not None else t[entry_col].min()
            te = pd.Timestamp(test_end) if test_end is not None else t[exit_col].max()
        else:
            ts = te = pd.Timestamp("2000-01-01")
    else:
        ts = pd.Timestamp(test_start)
        te = pd.Timestamp(test_end)

    test_bdays = max(int(pd.bdate_range(ts, te).size), 1)
    years = test_bdays / ppy

    rows = []
    for tk in universe:
        sub = t[t[asset_col] == tk]
        n_trades = int(len(sub))
        if n_trades == 0:
            rows.append(empty_row(tk))
            continue
        hold = sum(int(pd.bdate_range(r[entry_col], r[exit_col]).size) for _, r in sub.iterrows())
        rows.append(
            {
                "asset": tk,
                "n_trades": n_trades,
                "n_trades_per_year": float(n_trades / years) if years > 0 else 0.0,
                "total_hold_periods": int(hold),
                "pct_periods_in_market": float(min(hold / test_bdays, 1.0))
                if test_bdays > 0
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compare_to_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    clock: Clock = DAILY,
) -> dict:
    """Strategy vs benchmark metrics, deltas, information ratio and beta.

    Answers "did the strategy add anything over just holding the benchmark?" by putting
    both on the same per-period basis, then reporting the differences plus the
    active-return information ratio and the beta to the benchmark.
    """
    a = strategy_returns.dropna().astype(float)
    b = benchmark_returns.astype(float).reindex(a.index).fillna(0.0)
    sm = equity_metrics(a, clock=clock)
    bm = equity_metrics(b, clock=clock)

    active = a - b
    sqrt_ppy = clock.sharpe_factor()
    active_sd = float(active.std(ddof=1))
    ir = float(active.mean() / active_sd * sqrt_ppy) if active_sd > 0 else 0.0
    bvar = float(b.var(ddof=1))
    beta = float(np.cov(a.values, b.values, ddof=1)[0, 1] / bvar) if bvar > 0 else np.nan

    return {
        "strategy_cagr": sm["cagr"],
        "benchmark_cagr": bm["cagr"],
        "strategy_sharpe": sm["sharpe"],
        "benchmark_sharpe": bm["sharpe"],
        "strategy_calmar": sm["calmar"],
        "benchmark_calmar": bm["calmar"],
        "strategy_mdd": sm["mdd"],
        "benchmark_mdd": bm["mdd"],
        "delta_cagr": sm["cagr"] - bm["cagr"],
        "delta_sharpe": sm["sharpe"] - bm["sharpe"],
        "delta_calmar": sm["calmar"] - bm["calmar"],
        "delta_mdd": sm["mdd"] - bm["mdd"],
        "information_ratio": ir,
        "beta_to_benchmark": beta,
    }
