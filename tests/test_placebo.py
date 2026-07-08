"""Tests for the placebo module: generators + comparison check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import nullius as nl


def _autocorr_panel(phi: float = 0.6, n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, np.sqrt(1 - phi**2))
    return pd.DataFrame({"asset": ["A"] * n, "predicted": x})


def _lag1(s):
    x = np.asarray(s, dtype=float)
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


class TestGenerators:
    def test_uniform_preserves_shape_and_range(self):
        df = pd.DataFrame(
            {
                "asset": ["A"] * 100 + ["B"] * 100,
                "predicted": np.r_[np.linspace(0, 10, 100), np.linspace(-5, -1, 100)],
            }
        )
        out = nl.random_uniform_placebo(df, seed=42)
        a = out.loc[out["asset"] == "A", "predicted"]
        assert a.min() >= -1e-9 and a.max() <= 10 + 1e-9

    def test_uniform_breaks_autocorr_ar1_preserves(self):
        df = _autocorr_panel()
        assert _lag1(df["predicted"]) > 0.4
        ar1 = nl.ar1_matched_placebo(df, seed=0)
        uni = nl.random_uniform_placebo(df, seed=0)
        assert abs(_lag1(ar1["predicted"]) - _lag1(df["predicted"])) < 0.15
        assert abs(_lag1(uni["predicted"])) < 0.15

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="asset"):
            nl.random_uniform_placebo(pd.DataFrame({"predicted": [1.0, 2.0]}))
        with pytest.raises(ValueError, match="predicted"):
            nl.ar1_matched_placebo(pd.DataFrame({"asset": ["A"]}))

    def test_shuffled_entries_preserves_count_and_lengths(self):
        grid = pd.date_range("2021-01-01", periods=200, freq="D")
        trades = pd.DataFrame(
            {
                "asset": ["A", "A", "B"],
                "entry_time": [grid[5], grid[50], grid[10]],
                "exit_time": [grid[10], grid[60], grid[40]],
                "net_return": [0.01, -0.02, 0.05],
            }
        )
        out = nl.shuffled_entries_placebo(trades, grid, seed=1)
        assert len(out) == 3
        assert (out["asset"].value_counts() == trades["asset"].value_counts()).all()
        # holding length in bars preserved per trade (sorted by asset then entry)
        for asset in ("A", "B"):
            orig = trades[trades.asset == asset].sort_values("entry_time")
            new = out[out.asset == asset].sort_values("entry_time")
            orig_len = orig["exit_time"].values - orig["entry_time"].values
            new_len = new["exit_time"].values - new["entry_time"].values
            assert sorted(orig_len.tolist()) == sorted(new_len.tolist())

    def test_shuffled_entries_no_overlap(self):
        grid = pd.date_range("2021-01-01", periods=100, freq="D")
        trades = pd.DataFrame(
            {
                "asset": ["A"] * 3,
                "entry_time": [grid[0], grid[20], grid[40]],
                "exit_time": [grid[5], grid[25], grid[45]],
                "net_return": [0.0, 0.0, 0.0],
            }
        )
        out = nl.shuffled_entries_placebo(trades, grid, seed=2).sort_values("entry_time")
        exits = out["exit_time"].to_numpy()
        entries = out["entry_time"].to_numpy()
        assert (entries[1:] > exits[:-1]).all()

    def test_shuffled_entries_too_many_raises(self):
        grid = pd.date_range("2021-01-01", periods=5, freq="D")
        trades = pd.DataFrame(
            {
                "asset": ["A", "A"],
                "entry_time": [grid[0], grid[3]],
                "exit_time": [grid[2], grid[4]],
                "net_return": [0.0, 0.0],
            }
        )
        # 2 trades needing 3+2=5 bars, plus they must be non-overlapping in 5 → ok here;
        # shrink grid to force failure
        tiny = grid[:3]
        with pytest.raises(ValueError, match="non-overlapping"):
            nl.shuffled_entries_placebo(trades, tiny, seed=0)


class TestCompareAgainstPlacebos:
    def test_real_beats_placebos_passes(self):
        r = nl.compare_against_placebos(2.0, [0.1, -0.3, 0.2, 0.0, -0.1], quantile_threshold=0.95)
        assert r.passed is True
        assert r.statistic == 1.0

    def test_real_indistinguishable_fails(self):
        rng = np.random.default_rng(0)
        placebos = rng.normal(0, 1, 50)
        r = nl.compare_against_placebos(0.0, placebos)
        assert r.passed is False
        assert r.severity == "warning"

    def test_lower_is_better(self):
        r = nl.compare_against_placebos(0.01, [0.5, 0.6, 0.7, 0.8], higher_is_better=False)
        assert r.passed is True

    def test_needs_two_placebos(self):
        with pytest.raises(ValueError, match="at least 2"):
            nl.compare_against_placebos(1.0, [0.5])
