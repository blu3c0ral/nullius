"""QLIKE loss for variance forecasts — Patton (2011) robust loss.

QLIKE is robust to noise in the volatility proxy in a way MSE is not, which is why it is
the default loss for volatility-forecast comparison. **Trap:** the inputs are VARIANCE
(sigma^2), never volatility — square your vol series before passing it, or every number
here is silently wrong.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._common import ArrayLike
from ._common import to_1d_array as _to_1d_array


def qlike_loss(actual_var: ArrayLike, pred_var: ArrayLike) -> pd.Series:
    """Per-period QLIKE loss ``(actual/pred) - log(actual/pred) - 1`` on VARIANCE inputs.

    Detects a forecast that is systematically biased in variance terms, penalising
    under-prediction more than MSE would. Returns NaN where either input is NaN or
    ``pred <= 0`` / ``inf``; ``+inf`` where ``actual == 0`` (mathematically correct); NaN
    with a warning where ``actual < 0``.
    """
    index = None
    if isinstance(actual_var, pd.Series):
        index = actual_var.index
    elif isinstance(pred_var, pd.Series):
        index = pred_var.index

    a = _to_1d_array(actual_var, "actual_var")
    p = _to_1d_array(pred_var, "pred_var")

    if len(a) != len(p):
        raise ValueError(
            f"actual_var and pred_var must have the same length, got {len(a)} and {len(p)}"
        )

    n = len(a)
    bad_pred = (p <= 0) | np.isinf(p) | np.isnan(p)
    bad_actual = a < 0
    bad_either_nan = np.isnan(a) | np.isnan(p)

    n_bad_pred = int((bad_pred & ~np.isnan(p)).sum())
    n_bad_actual = int((bad_actual & ~np.isnan(a)).sum())
    if n_bad_pred > 0:
        warnings.warn(
            f"qlike_loss: {n_bad_pred} pred_var value(s) are <= 0 or inf; "
            "returning NaN for those elements.",
            RuntimeWarning,
            stacklevel=2,
        )
    if n_bad_actual > 0:
        warnings.warn(
            f"qlike_loss: {n_bad_actual} actual_var value(s) are < 0; "
            "returning NaN for those elements.",
            RuntimeWarning,
            stacklevel=2,
        )

    mask_nan = bad_pred | bad_either_nan | bad_actual
    valid = ~mask_nan
    ratio = np.where(valid, np.divide(a, p, where=valid, out=np.ones(n)), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        loss_vals = ratio - np.log(ratio) - 1.0
    out = np.where(mask_nan, np.nan, loss_vals)

    if index is not None:
        return pd.Series(out, index=index)
    return pd.Series(out)


def qlike_mean(actual_var: ArrayLike, pred_var: ArrayLike, *, skipna: bool = True) -> float:
    """Mean QLIKE loss over all observations; ``nan`` if no valid observations."""
    losses = qlike_loss(actual_var, pred_var)
    result = losses.mean(skipna=skipna)
    if pd.isna(result):
        return float(np.nan)
    return float(result)
