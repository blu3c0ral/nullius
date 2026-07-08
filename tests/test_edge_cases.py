"""Edge-case and branch-coverage tests for error paths and alternate options."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import nullius as nl
from nullius.forecast_eval import dm_stat
from nullius.forecast_eval._common import to_1d_array


class TestEquityEdges:
    def _prices(self, universe, n=60):
        rng = np.random.default_rng(0)
        dates = pd.bdate_range("2021-01-04", periods=n)
        rows = []
        for tk in universe:
            p = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
            for d, cl in zip(dates, p, strict=True):
                rows.append({"time": d, "asset": tk, "close": float(cl), "open": float(cl) * 0.999})
        return pd.DataFrame(rows), dates

    def test_empty_universe_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            nl.build_equity_curve(None, pd.DataFrame(), [])

    def test_overlapping_trades_raise(self):
        prices, dates = self._prices(["A"])
        trades = pd.DataFrame(
            {
                "asset": ["A", "A"],
                "entry_time": [dates[5], dates[8]],
                "exit_time": [dates[10], dates[12]],
                "exit_price": [101.0, 102.0],
                "net_return": [0.0, 0.0],
            }
        )
        with pytest.raises(ValueError, match="Overlapping"):
            nl.build_equity_curve(trades, prices, ["A"])

    def test_single_day_trade_and_cash_yield(self):
        prices, dates = self._prices(["A", "B"])
        trades = pd.DataFrame(
            {
                "asset": ["A"],
                "entry_time": [dates[5]],
                "exit_time": [dates[5]],  # single-day
                "exit_price": [100.5],
                "net_return": [0.005],
            }
        )
        cash = pd.Series(0.0001, index=pd.DatetimeIndex(sorted(prices["time"].unique())))
        dl = nl.build_equity_curve(trades, prices, ["A", "B"], cash_period_yield=cash)
        assert "portfolio_ret" in dl.columns

    def test_rf_period_series(self):
        idx = pd.date_range("2021-01-01", periods=200, freq="D")
        r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 200), index=idx)
        rf = pd.Series(np.full(200, 0.0002), index=idx)
        m_rf = nl.equity_metrics(r, rf_period_series=rf)["sharpe"]
        m_0 = nl.equity_metrics(r)["sharpe"]
        assert m_rf < m_0  # subtracting a positive rf lowers the Sharpe

    def test_exposure_empty_trades(self):
        out = nl.exposure_stats(pd.DataFrame(), ["A", "B"])
        assert (out["n_trades"] == 0).all()
        out2 = nl.exposure_stats(None, ["A"])
        assert out2["n_trades"].iloc[0] == 0

    def test_buy_and_hold_true(self):
        prices, _ = self._prices(["A", "B"])
        curve = nl.buy_and_hold_curve(prices, ["A", "B"], rebalanced=False)
        assert {"time", "bh_ret", "bh_equity"} <= set(curve.columns)

    def test_compare_zero_variance_benchmark(self):
        idx = pd.date_range("2021-01-01", periods=50, freq="D")
        strat = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 50), index=idx)
        bench = pd.Series(np.zeros(50), index=idx)  # zero variance
        out = nl.compare_to_benchmark(strat, bench)
        assert np.isnan(out["beta_to_benchmark"])


class TestForecastEdges:
    def test_to_1d_array_rejects_2d(self):
        with pytest.raises(ValueError, match="1-D"):
            to_1d_array(np.zeros((3, 2)), "x")

    def test_qlike_negative_actual_warns(self):
        with pytest.warns(RuntimeWarning):
            loss = nl.qlike_loss(np.array([-0.01, 0.04]), np.array([0.04, 0.04]))
        assert np.isnan(loss.iloc[0])

    def test_dm_stat_tie(self):
        res = dm_stat(np.full(50, 0.5))
        assert res.dm_stat == 0.0 and res.dm_p_value == 1.0

    def test_dm_mse_mae_and_callable(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 100)
        p1 = a + rng.normal(0, 0.1, 100)
        p2 = a + rng.normal(0, 0.5, 100)
        assert nl.diebold_mariano(a, p1, p2, loss="mse")["better_model"] == "pred1"
        assert nl.diebold_mariano(a, p1, p2, loss="mae")["better_model"] == "pred1"
        out = nl.diebold_mariano(a, p1, p2, loss=lambda y, f: (y - f) ** 2)
        assert out["loss_name"] == "custom"

    def test_dm_unknown_loss_and_bad_callable(self):
        a = np.random.default_rng(0).normal(0, 1, 50)
        with pytest.raises(ValueError, match="Unknown loss"):
            nl.diebold_mariano(a, a, a, loss="zzz")
        with pytest.raises(ValueError, match="1-D array"):
            nl.diebold_mariano(a, a, a, loss=lambda y, f: np.zeros((2, 2)))

    def test_dm_test_no_asset_column(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {"time": pd.date_range("2021", periods=100), "loss_diff": rng.normal(0.05, 0.1, 100)}
        )
        cr = nl.dm_test(frame)
        assert cr.name == "diebold_mariano"

    def test_dm_test_missing_loss_diff(self):
        with pytest.raises(ValueError, match="loss_diff"):
            nl.dm_test(pd.DataFrame({"time": [1, 2]}))

    def test_mz_hac0_and_near_perfect(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.02, 0.06, 100)
        a = p + rng.normal(0, 0.003, 100)
        out = nl.mincer_zarnowitz(a, p, hac_lag=0)
        assert out["hac_lag_used"] == 0
        # near-perfect (degenerate residual): H0 satisfied → wald ~0, p ~1
        perfect = nl.mincer_zarnowitz(p, p)
        assert perfect["wald_p_value"] == pytest.approx(1.0)

    def test_mz_degenerate_but_biased(self):
        p = np.linspace(0.02, 0.06, 100)
        a = p + 0.01  # constant bias, zero-variance residual
        out = nl.mincer_zarnowitz(a, p)
        assert out["wald_p_value"] < 0.05

    def test_calibrate_linear_space(self):
        rng = np.random.default_rng(0)
        n = 600
        df = pd.DataFrame(
            {
                "predicted": rng.normal(0.2, 0.05, n),
                "actual": rng.normal(0.2, 0.05, n),
                "fold": np.repeat([0, 1, 2], n // 3),
                "horizon": 20,
            }
        )
        out = nl.calibrate_forecasts_rolling(df, min_prior_obs=150, log_space=False)
        assert out.loc[out["fold"] == 2, "calibrated"].all()


class TestOtherEdges:
    def test_tsv_single_value(self):
        assert nl.trial_sharpe_variance([1.0]) == 0.0

    def test_ar1_constant_series(self):
        df = pd.DataFrame({"asset": ["A"] * 50, "predicted": np.full(50, 3.0)})
        out = nl.ar1_matched_placebo(df, seed=0)
        assert (out["predicted"] == 3.0).all()

    def test_compare_placebos_zero_variance(self):
        cr = nl.compare_against_placebos(1.0, [0.0, 0.0, 0.0, 0.0])
        assert cr.passed is True  # beats all

    def test_tracked_scalar_return(self):
        reg = nl.TrialRegistry()

        @reg.tracked(name="scan")
        def bt(x):
            return x * 2  # scalar, not a mapping

        bt(x=3)
        assert reg.to_frame()["metrics.value"].iloc[0] == 6

    def test_subperiod_pooled(self):
        rng = np.random.default_rng(0)
        times = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "time": times,
                "asset": "A",
                "predicted": rng.normal(0, 1, 100),
                "actual": rng.normal(0, 1, 100),
            }
        )
        out = nl.subperiod_ic_stability(df, {"all": ("2020-01-01", "2020-12-31")}, per_asset=False)
        assert out["asset"].iloc[0] == "ALL"

    def test_prefix_duplicate_keys_flagged(self):
        # a pipeline that emits duplicate (time, asset) keys cannot be aligned → fatal
        data = pd.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=40, freq="D"),
                "predicted": np.arange(40.0),
            }
        )

        def dup_pipeline(df):
            out = df[["time"]].copy()
            out["signal"] = 1.0
            return pd.concat([out, out], ignore_index=True)  # duplicated rows

        r = nl.check_prefix_invariance(dup_pipeline, data)
        assert r.passed is False

    def test_prefix_membership_change(self):
        # pipeline that drops early rows only when it sees the full sample → membership diff
        data = pd.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=60, freq="D"),
                "predicted": np.arange(60.0),
            }
        )

        def flaky(df):
            out = df[["time"]].copy()
            out["signal"] = 1.0
            if len(df) == 60:  # only on full data, drop the first row
                out = out.iloc[1:]
            return out

        r = nl.check_prefix_invariance(flaky, data)
        assert r.passed is False
