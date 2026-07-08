"""Tests for the metrics module: equity metrics + Sharpe inference ([M2] property tests)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import nullius as nl
from nullius import DAILY, Clock


class TestEquityMetrics:
    def test_empty_series(self):
        m = nl.equity_metrics(pd.Series([], dtype=float))
        assert m["sharpe"] == 0.0 and m["n_periods"] == 0

    def test_positive_drift_positive_sharpe(self):
        idx = pd.date_range("2021-01-01", periods=252, freq="D")
        r = pd.Series(np.full(252, 0.0004), index=idx)
        m = nl.equity_metrics(r)
        assert m["sharpe"] > 0
        assert m["mdd"] == pytest.approx(0.0, abs=1e-9)
        assert m["cagr"] > 0

    def test_wipeout_cagr_minus_one(self):
        idx = pd.date_range("2021-01-01", periods=10, freq="D")
        r = pd.Series([-1.0] + [0.0] * 9, index=idx)  # -100% day 1
        m = nl.equity_metrics(r)
        assert m["cagr"] == -1.0

    def test_clock_changes_annualization(self):
        idx = pd.date_range("2021-01-01", periods=500, freq="D")
        r = pd.Series(np.random.default_rng(0).normal(0.0002, 0.01, 500), index=idx)
        daily = nl.equity_metrics(r, clock=DAILY)["sharpe"]
        intraday = nl.equity_metrics(r, clock=Clock(periods_per_year=252 * 78))["sharpe"]
        # Sharpe scales by the sqrt-time factor ratio, regardless of sign.
        assert intraday == pytest.approx(daily * np.sqrt(78), rel=1e-12)

    def test_rf_reduces_sharpe(self):
        idx = pd.date_range("2021-01-01", periods=252, freq="D")
        r = pd.Series(np.full(252, 0.0004), index=idx)
        s0 = nl.equity_metrics(r, rf_annual=0.0)["sharpe"]
        s5 = nl.equity_metrics(r, rf_annual=0.05)["sharpe"]
        assert s5 < s0

    def test_compare_to_benchmark(self):
        idx = pd.date_range("2021-01-01", periods=252, freq="D")
        rng = np.random.default_rng(1)
        strat = pd.Series(rng.normal(0.0006, 0.01, 252), index=idx)
        bench = pd.Series(rng.normal(0.0002, 0.01, 252), index=idx)
        out = nl.compare_to_benchmark(strat, bench)
        assert "delta_sharpe" in out and "information_ratio" in out and "beta_to_benchmark" in out


class TestSharpeInference:
    def test_sharpe_se_units(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.0005, 0.01, 1000)
        se_ann = nl.sharpe_se(r, clock=DAILY, annualized=True)
        se_pp = nl.sharpe_se(r, clock=DAILY, annualized=False)
        assert se_ann == pytest.approx(se_pp * np.sqrt(252), rel=1e-12)
        assert se_pp == pytest.approx(
            np.sqrt((1 + 0.5 * (r.mean() / r.std(ddof=1)) ** 2) / 1000), rel=1e-9
        )

    def test_sharpe_se_nan_short(self):
        assert np.isnan(nl.sharpe_se([0.01]))

    def test_property_size(self):
        # [M2] size: two iid series with EQUAL true Sharpe reject at ~ alpha, not more.
        rng = np.random.default_rng(0)
        rejections = 0
        trials = 200
        for _ in range(trials):
            a = rng.normal(0.0005, 0.01, 300)
            b = rng.normal(0.0005, 0.01, 300)
            if nl.sharpe_difference_test(a, b, alpha=0.05).passed:
                rejections += 1
        assert rejections / trials < 0.15  # nominal 0.05; guards against gross over-rejection

    def test_property_power(self):
        # [M2] power: a genuine Sharpe gap is detected most of the time.
        rng = np.random.default_rng(1)
        detected = 0
        trials = 50
        for _ in range(trials):
            a = rng.normal(0.0015, 0.01, 500)  # higher Sharpe
            b = rng.normal(0.0000, 0.01, 500)
            if nl.sharpe_difference_test(a, b, alpha=0.05).passed:
                detected += 1
        assert detected / trials > 0.6

    def test_bootstrap_method_runs(self):
        rng = np.random.default_rng(2)
        a = rng.normal(0.001, 0.01, 300)
        b = rng.normal(0.0, 0.01, 300)
        cr = nl.sharpe_difference_test(a, b, method="bootstrap", n_boot=300, seed=0)
        assert cr.name == "sharpe_difference"
        assert cr.severity == "info"

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            nl.sharpe_difference_test([0.1, 0.2], [0.1, 0.2, 0.3])

    def test_bad_method(self):
        with pytest.raises(ValueError, match="method"):
            nl.sharpe_difference_test([0.1] * 10, [0.2] * 10, method="zzz")
