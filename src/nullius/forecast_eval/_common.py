"""Shared helpers for the forecast-evaluation module."""

from __future__ import annotations

import numpy as np
import pandas as pd

ArrayLike = pd.Series | np.ndarray


def to_1d_array(x: ArrayLike, name: str) -> np.ndarray:
    """Coerce a Series/array to a 1-D float ndarray (NaN-preserving)."""
    if isinstance(x, pd.Series):
        arr = x.to_numpy(dtype=float, na_value=np.nan)
    else:
        arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


def default_hac_lag(n_obs: int, horizon: int) -> int:
    """Newey-West (1994) plug-in truncation lag, floored at ``horizon - 1``."""
    base = int(np.floor(4 * (n_obs / 100) ** (2 / 9)))
    return max(horizon - 1, base)
