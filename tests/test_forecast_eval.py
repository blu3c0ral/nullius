"""Tests for forecast evaluation: QLIKE, MZ (+statsmodels cross-check), DM+panel, calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import nullius as nl
from nullius.forecast_eval import dm_stat


class TestQlike:
    def test_perfect_forecast_zero_loss(self):
        a = np.array([0.04, 0.02, 0.09, 0.05])
        loss = nl.qlike_loss(a, a)
        np.testing.assert_allclose(loss.to_numpy(), 0.0, atol=1e-12)

    def test_bad_pred_returns_nan_with_warning(self):
        with pytest.warns(RuntimeWarning):
            loss = nl.qlike_loss(np.array([0.04, 0.04]), np.array([-1.0, 0.04]))
        assert np.isnan(loss.iloc[0]) and not np.isnan(loss.iloc[1])

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            nl.qlike_loss([0.1, 0.2], [0.1])

    def test_qlike_mean_skips_nan(self):
        with pytest.warns(RuntimeWarning):
            m = nl.qlike_mean(np.array([0.04, 0.04]), np.array([-1.0, 0.04]))
        assert m == pytest.approx(0.0, abs=1e-9)


class TestMincerZarnowitz:
    def _data(self, bias=0.0, n=400, seed=0):
        # MZ-efficient forecast: realised = forecast + independent error, so regressing
        # actual on predicted gives alpha=0, beta=1. A non-zero bias breaks alpha=0.
        rng = np.random.default_rng(seed)
        predicted = rng.uniform(0.02, 0.06, n)
        actual = bias + predicted + rng.normal(0, 0.004, n)
        return actual, predicted

    def test_unbiased_not_rejected(self):
        a, p = self._data(bias=0.0)
        out = nl.mincer_zarnowitz(a, p)
        assert out["wald_p_value"] > 0.05
        assert out["beta"] == pytest.approx(1.0, abs=0.15)

    def test_biased_rejected(self):
        a, p = self._data(bias=0.02)
        out = nl.mincer_zarnowitz(a, p)
        assert out["wald_p_value"] < 0.05

    def test_too_few_obs(self):
        with pytest.raises(ValueError, match="at least 30"):
            nl.mincer_zarnowitz(np.arange(10.0), np.arange(10.0))

    def test_constant_predictor(self):
        with pytest.raises(ValueError, match="constant"):
            nl.mincer_zarnowitz(np.random.default_rng(0).normal(size=40), np.full(40, 0.05))

    def test_crosscheck_against_statsmodels(self):
        sm = pytest.importorskip("statsmodels.api")
        a, p = self._data(bias=0.0, n=500, seed=3)
        L = 5
        out = nl.mincer_zarnowitz(a, p, hac_lag=L)
        X = sm.add_constant(p)
        res = sm.OLS(a, X).fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": True})
        assert out["alpha"] == pytest.approx(float(res.params[0]), rel=1e-9)
        assert out["beta"] == pytest.approx(float(res.params[1]), rel=1e-9)
        assert out["alpha_se"] == pytest.approx(float(res.bse[0]), rel=1e-8)
        assert out["beta_se"] == pytest.approx(float(res.bse[1]), rel=1e-8)


class TestDieboldMariano:
    def _data(self, n=300, seed=0):
        rng = np.random.default_rng(seed)
        actual = np.abs(rng.normal(0.04, 0.01, n)) + 0.001
        good = actual * (1 + rng.normal(0, 0.05, n))
        bad = actual * (1 + rng.normal(0, 0.3, n)) + 0.01
        return actual, good, bad

    def test_identical_forecasts_tie(self):
        a, g, _ = self._data()
        out = nl.diebold_mariano(a, g, g, loss="qlike")
        assert out["better_model"] == "tie"
        assert out["dm_stat"] == 0.0

    def test_better_model_detected(self):
        a, g, b = self._data()
        out = nl.diebold_mariano(a, g, b, loss="qlike")
        assert out["better_model"] == "pred1"
        assert out["dm_stat"] < 0

    def test_dm_stat_kernel_hac0_matches_manual(self):
        rng = np.random.default_rng(0)
        d = rng.normal(0.1, 1.0, 200)
        res = dm_stat(d, hac_lag=0)
        manual = d.mean() / np.sqrt(d.var(ddof=0) / len(d))
        assert res.dm_stat == pytest.approx(manual, rel=1e-12)

    def test_dm_stat_too_few(self):
        with pytest.raises(ValueError, match="at least 2"):
            dm_stat([1.0])

    def test_too_few_obs(self):
        with pytest.raises(ValueError, match="at least 30"):
            nl.diebold_mariano(np.arange(10.0), np.arange(10.0), np.arange(10.0), loss="mse")

    def test_panel_deflates_vs_pooled(self):
        # Correlated cross-section: pooling N assets inflates |stat| vs time-clustering.
        rng = np.random.default_rng(0)
        n_t, n_a = 150, 8
        times = np.repeat(pd.date_range("2021-01-01", periods=n_t, freq="D"), n_a)
        assets = np.tile([f"A{i}" for i in range(n_a)], n_t)
        common = np.repeat(rng.normal(0.02, 0.05, n_t), n_a)  # shared per-time signal
        loss_diff = common + rng.normal(0, 0.05, n_t * n_a)
        frame = pd.DataFrame({"time": times, "asset": assets, "loss_diff": loss_diff})

        panel = nl.diebold_mariano_panel(frame)
        pooled = dm_stat(frame["loss_diff"].to_numpy())
        assert abs(panel.dm_stat) < abs(pooled.dm_stat)

    def test_dm_test_dispatch_and_forced_pool_warning(self):
        rng = np.random.default_rng(1)
        n_t, n_a = 120, 5
        times = np.repeat(pd.date_range("2021-01-01", periods=n_t, freq="D"), n_a)
        assets = np.tile([f"A{i}" for i in range(n_a)], n_t)
        loss_diff = rng.normal(0.01, 0.1, n_t * n_a)
        frame = pd.DataFrame({"time": times, "asset": assets, "loss_diff": loss_diff})

        auto = nl.dm_test(frame)
        assert auto.severity == "info"
        forced = nl.dm_test(frame, cluster=False)
        assert forced.severity == "warning"
        assert "pooled" in forced.details.lower()


class TestCalibration:
    def test_reliability_table_perfect(self):
        rng = np.random.default_rng(0)
        pred = rng.uniform(0, 1, 500)
        actual = pred + rng.normal(0, 0.01, 500)
        tbl = nl.reliability_table(pred, actual, n_bins=5)
        assert len(tbl) == 5
        np.testing.assert_allclose(tbl["mean_predicted"], tbl["mean_actual"], atol=0.05)

    def test_reliability_table_empty(self):
        tbl = nl.reliability_table([], [])
        assert tbl.empty

    def test_calibrate_missing_columns(self):
        with pytest.raises(ValueError, match="missing columns"):
            nl.calibrate_forecasts_rolling(pd.DataFrame({"predicted": [1.0]}))

    def test_calibrate_early_fold_passthrough(self):
        # First fold has no prior history → identity, calibrated=False.
        rng = np.random.default_rng(0)
        n = 400
        df = pd.DataFrame(
            {
                "predicted": np.abs(rng.normal(0.2, 0.05, n)) + 0.01,
                "actual": np.abs(rng.normal(0.2, 0.05, n)) + 0.01,
                "fold": np.repeat([0, 1], n // 2),
                "horizon": 20,
            }
        )
        out = nl.calibrate_forecasts_rolling(df, min_prior_obs=250)
        assert not out.loc[out["fold"] == 0, "calibrated"].any()
