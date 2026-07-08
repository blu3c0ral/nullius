"""Shared test fixtures and the golden-parity loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GOLDEN = Path(__file__).parent / "golden"


def load_golden(name: str) -> dict:
    with (GOLDEN / f"{name}.json").open() as fh:
        return json.load(fh)


@pytest.fixture
def golden():
    return load_golden


@pytest.fixture
def predictions_panel() -> pd.DataFrame:
    """A small (time, asset, predicted, actual) panel with mild predictive signal."""
    rng = np.random.default_rng(0)
    times = pd.date_range("2021-01-01", periods=120, freq="D")
    frames = []
    for asset in ("AAA", "BBB", "CCC"):
        actual = rng.normal(0, 1, len(times))
        predicted = 0.3 * actual + rng.normal(0, 1, len(times))
        frames.append(
            pd.DataFrame({"time": times, "asset": asset, "predicted": predicted, "actual": actual})
        )
    return pd.concat(frames, ignore_index=True)
