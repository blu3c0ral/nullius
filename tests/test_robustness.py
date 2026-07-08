"""Tests for the robustness module: bootstrap IC p-value + subperiod stratification."""

from __future__ import annotations

import numpy as np
import pandas as pd

import nullius as nl
from nullius import US_EQUITY_REGIMES


class TestBootstrap:
    def test_no_signal_high_pvalue(self):
        rng = np.random.default_rng(0)
        pred = pd.Series(rng.normal(0, 1, 300))
        act = pd.Series(rng.normal(0, 1, 300))  # independent
        out = nl.ic_bootstrap_pvalue(pred, act, n_bootstrap=500, seed=1)
        assert out["p_value"] > 0.05

    def test_strong_signal_low_pvalue(self):
        rng = np.random.default_rng(0)
        pred = pd.Series(rng.normal(0, 1, 300))
        act = pred * 0.5 + pd.Series(rng.normal(0, 1, 300))
        out = nl.ic_bootstrap_pvalue(pred, act, n_bootstrap=500, seed=1)
        assert out["p_value"] < 0.05
        assert out["observed_ic"] > 0.2

    def test_too_short(self):
        out = nl.ic_bootstrap_pvalue(pd.Series([1.0]), pd.Series([1.0]))
        assert out["p_value"] == 1.0

    def test_fisher_combines(self):
        out = nl.fisher_combined_pvalue(pd.Series([0.01, 0.02, 0.5]))
        assert out["n_assets"] == 3
        assert out["combined_p"] < 0.05

    def test_fisher_all_nan(self):
        out = nl.fisher_combined_pvalue(pd.Series([np.nan, np.nan]))
        assert out["combined_p"] == 1.0 and out["n_assets"] == 0


class TestSubperiod:
    def _panel(self, seed=0):
        rng = np.random.default_rng(seed)
        times = np.tile(pd.date_range("2020-01-01", periods=200, freq="D"), 2)
        assets = np.repeat(["A", "B"], 200)
        pred = rng.normal(0, 1, 400)
        actual = pred * 0.3 + rng.normal(0, 1, 400)
        return pd.DataFrame({"time": times, "asset": assets, "predicted": pred, "actual": actual})

    def test_subperiod_ic_per_asset(self):
        df = self._panel()
        periods = {"h1": ("2020-01-01", "2020-04-01"), "h2": ("2020-04-02", "2020-07-01")}
        out = nl.subperiod_ic_stability(df, periods)
        assert set(out["period"]) <= {"h1", "h2"}
        assert set(out["asset"]) == {"A", "B"}

    def test_min_obs_filter(self):
        df = self._panel()
        # a period with too few obs should be dropped
        periods = {"tiny": ("2020-01-01", "2020-01-05")}
        out = nl.subperiod_ic_stability(df, periods, min_obs=20)
        assert out.empty

    def test_regime_stratified_metrics(self):
        idx = pd.date_range("2020-01-01", periods=600, freq="D")
        r = pd.Series(np.random.default_rng(0).normal(0.0003, 0.01, 600), index=idx)
        out = nl.regime_stratified_metrics(r, US_EQUITY_REGIMES)
        assert "sharpe" in out.columns and "calmar" in out.columns and "period" in out.columns
        assert len(out) > 0

    def test_example_regimes_shape(self):
        assert "covid_crash" in US_EQUITY_REGIMES
        assert all(len(v) == 2 for v in US_EQUITY_REGIMES.values())
