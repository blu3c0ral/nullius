"""Tests for the leakage module: prefix-invariance, purged splits, negative controls.

Includes the signature move — every probe has a test asserting it FIRES on its paired
leaky reference implementation, and the [M1] prefix-invariance property test (leaky flagged
on 100/100 seeds, clean passes on 100/100).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import nullius as nl
from nullius.leakage import Split


def _panel(seed: int, n_times: int = 150, assets=("A", "B")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2020-01-01", periods=n_times, freq="D")
    frames = []
    for a in assets:
        frames.append(
            pd.DataFrame({"time": times, "asset": a, "predicted": rng.normal(0, 1, n_times)})
        )
    return pd.concat(frames, ignore_index=True)


class TestPrefixInvariance:
    def test_leaky_gate_is_flagged(self):
        r = nl.check_prefix_invariance(nl.leaky_global_quantile_gate, _panel(0))
        assert r.passed is False
        assert r.severity == "fatal"
        assert r.statistic > 0
        assert r.evidence is not None and len(r.evidence) > 0

    def test_clean_gate_passes(self):
        r = nl.check_prefix_invariance(nl.clean_expanding_quantile_gate, _panel(0))
        assert r.passed is True
        assert r.statistic == 0.0
        assert r.evidence is None

    def test_property_leaky_flagged_100_of_100(self):
        # [M1] property test
        flagged = sum(
            nl.check_prefix_invariance(nl.leaky_global_quantile_gate, _panel(s)).passed is False
            for s in range(100)
        )
        assert flagged == 100

    def test_property_clean_passes_100_of_100(self):
        passed = sum(
            nl.check_prefix_invariance(nl.clean_expanding_quantile_gate, _panel(s)).passed is True
            for s in range(100)
        )
        assert passed == 100

    def test_single_asset_no_asset_key(self):
        data = _panel(1, assets=("A",))
        r = nl.check_prefix_invariance(nl.clean_expanding_quantile_gate, data)
        assert r.passed is True

    def test_custom_cut_times(self):
        data = _panel(2)
        cuts = [pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-01")]
        r = nl.check_prefix_invariance(nl.leaky_global_quantile_gate, data, cut_times=cuts)
        assert r.passed is False

    def test_missing_time_col_raises(self):
        with pytest.raises(ValueError, match="time_col"):
            nl.check_prefix_invariance(lambda d: d, pd.DataFrame({"x": [1]}))

    def test_pipeline_without_time_col_raises(self):
        data = _panel(0)
        with pytest.raises(ValueError, match="time_col"):
            nl.check_prefix_invariance(lambda d: d[["asset"]], data)


class TestSplitPurging:
    def test_purged_splits_are_clean(self):
        splits = nl.purged_splits(range(500), train_span=100, test_span=50, label_horizon=20)
        assert len(splits) > 0
        r = nl.check_split_purging(splits, label_horizon=20)
        assert r.passed is True

    def test_leaky_unpurged_splits_flagged(self):
        splits = nl.leaky_unpurged_splits(range(500), train_span=100, test_span=50)
        r = nl.check_split_purging(splits, label_horizon=20)
        assert r.passed is False
        assert r.severity == "fatal"
        assert r.evidence is not None and len(r.evidence) == len(splits)

    def test_check_accepts_raw_tuples(self):
        pairs = [(np.arange(0, 80), np.arange(100, 120))]  # gap 20
        assert nl.check_split_purging(pairs, label_horizon=20).passed is True
        pairs_leaky = [(np.arange(0, 100), np.arange(100, 120))]  # adjacent
        assert nl.check_split_purging(pairs_leaky, label_horizon=20).passed is False

    def test_expanding_splits(self):
        rolling = nl.purged_splits(range(400), train_span=50, test_span=25, label_horizon=5)
        expanding = nl.purged_splits(
            range(400), train_span=50, test_span=25, label_horizon=5, expanding=True
        )
        # expanding trains on everything from 0; rolling caps at train_span
        assert expanding[-1].train_idx.size > rolling[-1].train_idx.size
        assert nl.check_split_purging(expanding, label_horizon=5).passed is True

    def test_split_dataclass(self):
        s = Split(train_idx=np.arange(10), test_idx=np.arange(15, 20))
        assert s.train_idx.max() == 9

    def test_invalid_params(self):
        with pytest.raises(ValueError, match="train_span"):
            nl.purged_splits(range(100), train_span=0, test_span=10, label_horizon=1)
        with pytest.raises(ValueError, match="non-negative"):
            nl.purged_splits(range(100), train_span=10, test_span=10, label_horizon=-1)
