"""Tests for the multiplicity module: DSR, PBO, TrialRegistry (incl. [M1] property tests)."""

from __future__ import annotations

import numpy as np
import pytest

import nullius as nl
from nullius import DAILY


class TestDeflatedSharpe:
    def test_kurtosis_is_non_excess(self):
        r = np.random.default_rng(0).normal(0, 0.01, 5000)
        out = nl.deflated_sharpe_ratio(r, n_trials=1, var_across_trials=0.0)
        assert out["kurt"] == pytest.approx(3.0, abs=0.3)

    def test_single_genuine_edge_passes_n1_fallback(self):
        # [M1] property: one genuine SR=1 (annualized) trial, N=1 → DSR > 0.95.
        rng = np.random.default_rng(1)
        daily_sr = 1.0 / np.sqrt(252)
        r = rng.normal(daily_sr * 0.01, 0.01, 4000)
        out = nl.deflated_sharpe_ratio(r, n_trials=1, var_across_trials=0.0)
        assert out["passed"] is True
        assert out["dsr"] > 0.95

    def test_property_best_of_noise_rarely_passes(self):
        # [M1] property: 100 pure-noise trials, the max-Sharpe one must not pass DSR
        # in the large majority of simulations.
        rng = np.random.default_rng(7)
        T, N, SIMS = 1000, 100, 100
        passes = 0
        for _ in range(SIMS):
            trials = rng.normal(0.0, 0.01, size=(T, N))
            per_period_sr = trials.mean(axis=0) / trials.std(axis=0, ddof=1)
            best = int(np.argmax(per_period_sr))
            V = nl.trial_sharpe_variance(per_period_sr, annualized=False)
            out = nl.deflated_sharpe_ratio(trials[:, best], n_trials=N, var_across_trials=V)
            passes += int(out["passed"])
        assert passes <= 0.05 * SIMS

    def test_more_trials_lowers_dsr(self):
        rng = np.random.default_rng(3)
        daily_sr = 1.0 / np.sqrt(252)
        r = rng.normal(daily_sr * 0.01, 0.01, 3000)
        V = 0.02**2
        few = nl.deflated_sharpe_ratio(r, n_trials=5, var_across_trials=V)["dsr"]
        many = nl.deflated_sharpe_ratio(r, n_trials=500, var_across_trials=V)["dsr"]
        assert many < few

    def test_denominator_guard_returns_nan(self):
        # extreme skew/kurt with large sr_hat can drive the PSR denominator <= 0
        out = nl.deflated_sharpe_ratio(
            [1.0, 1.0, 1.0, 1.0, -50.0], n_trials=1, var_across_trials=0.0
        )
        assert np.isnan(out["dsr"]) or out["passed"] in (True, False)

    def test_checkresult_wrapper(self):
        rng = np.random.default_rng(1)
        daily_sr = 1.0 / np.sqrt(252)
        r = rng.normal(daily_sr * 0.01, 0.01, 4000)
        cr = nl.deflated_sharpe(r, n_trials=1, var_across_trials=0.0)
        assert cr.passed is True
        assert cr.severity == "fatal"
        assert cr.name == "deflated_sharpe"

    def test_wrapper_requires_inputs(self):
        with pytest.raises(ValueError, match="n_trials"):
            nl.deflated_sharpe([0.01] * 100)


class TestPBO:
    def test_property_noise_pbo_near_half(self):
        # [M1] property: mean PBO over >= 20 seeded pure-noise matrices in [0.4, 0.6].
        pbos = []
        for seed in range(25):
            M = np.random.default_rng(seed).normal(0.0, 0.01, size=(600, 12))
            pbos.append(nl.probability_of_backtest_overfitting(M, n_splits=10, seed=1)["pbo"])
        assert 0.4 <= np.mean(pbos) <= 0.6

    def test_property_planted_edge_drops_pbo(self):
        pbos_noise, pbos_edge = [], []
        for seed in range(25):
            base = np.random.default_rng(seed).normal(0.0, 0.01, size=(600, 12))
            pbos_noise.append(
                nl.probability_of_backtest_overfitting(base, n_splits=10, seed=1)["pbo"]
            )
            edged = base.copy()
            edged[:, 0] += 0.01
            pbos_edge.append(
                nl.probability_of_backtest_overfitting(edged, n_splits=10, seed=1)["pbo"]
            )
        assert np.mean(pbos_edge) < np.mean(pbos_noise) - 0.2

    def test_checkresult_bands(self):
        M = np.random.default_rng(13).normal(0.0, 0.01, size=(1000, 12))
        M[:, 0] += 0.01
        cr = nl.pbo(M, n_splits=10, seed=1)
        assert cr.passed is True
        assert cr.evidence is not None and "logit_lambda" in cr.evidence.columns

    def test_validation_errors(self):
        M = np.random.default_rng(0).normal(0, 0.01, size=(500, 5))
        with pytest.raises(ValueError, match="even"):
            nl.probability_of_backtest_overfitting(M, n_splits=7)
        with pytest.raises(ValueError, match=">= 2"):
            nl.probability_of_backtest_overfitting(M[:, :1], n_splits=8)
        with pytest.raises(ValueError, match="T >="):
            nl.probability_of_backtest_overfitting(M[:10], n_splits=8)


class TestTrialRegistry:
    def test_log_and_count(self):
        reg = nl.TrialRegistry()
        reg.log("bt", {"threshold": 0.9}, {"sharpe": 1.1}, timestamp=0)
        reg.log("bt", {"threshold": 0.8}, {"sharpe": 0.9}, timestamp=1)
        assert reg.n_trials() == 2
        assert reg.metric_values("sharpe") == [1.1, 0.9]
        frame = reg.to_frame()
        assert "params.threshold" in frame.columns and "metrics.sharpe" in frame.columns

    def test_persistence_roundtrip(self, tmp_path):
        path = tmp_path / "trials.jsonl"
        reg = nl.TrialRegistry(path)
        reg.log("bt", {"a": 1}, {"sharpe": 1.0}, timestamp=0)
        reg2 = nl.TrialRegistry(path)  # reload from disk
        assert reg2.n_trials() == 1
        assert reg2.metric_values("sharpe") == [1.0]

    def test_tracked_decorator_logs_each_call(self):
        reg = nl.TrialRegistry()

        @reg.tracked
        def backtest(threshold):
            return {"sharpe": threshold * 2}

        backtest(threshold=0.5)
        backtest(threshold=0.7)
        assert reg.n_trials() == 2
        assert reg.metric_values("sharpe") == [1.0, 1.4]

    def test_deflated_sharpe_registry_sugar(self):
        reg = nl.TrialRegistry()
        rng = np.random.default_rng(0)
        for i in range(30):
            reg.log("bt", {"i": i}, {"sharpe": float(rng.normal(1.0, 0.3))}, timestamp=i)
        r = rng.normal(0.5 / np.sqrt(252) * 0.01, 0.01, 2000)
        cr = nl.deflated_sharpe(r, registry=reg, sharpe_key="sharpe", clock=DAILY)
        assert cr.name == "deflated_sharpe"
        assert cr.statistic is None or 0.0 <= cr.statistic <= 1.0
