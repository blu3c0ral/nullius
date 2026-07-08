"""Forecast calibration: leakage-free rolling recalibration + a reliability table."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ols_ab(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Intercept and slope of ``y ~ a + b*x`` via least squares."""
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


def calibrate_forecasts_rolling(
    predictions: pd.DataFrame,
    *,
    pred_col: str = "predicted",
    actual_col: str = "actual",
    fold_col: str = "fold",
    horizon_col: str = "horizon",
    out_col: str = "predicted_calibrated",
    min_prior_obs: int = 250,
    log_space: bool = True,
) -> pd.DataFrame:
    """Leakage-free rolling Mincer-Zarnowitz recalibration of forecasts.

    Fixes level miscalibration (MZ ``beta != 1``) without touching the model. For each
    horizon, folds are processed in chronological order; the map ``actual ~ a + b*predicted``
    (fit in log space by default) is estimated on the pooled out-of-sample predictions of
    all **earlier** folds and applied to the current fold. Because fold ``f`` uses only
    folds ``< f``, there is no look-ahead. Early folds with fewer than ``min_prior_obs``
    prior rows pass through uncalibrated (identity), flagged by the returned ``calibrated``
    column.

    Returns a copy of ``predictions`` with two added columns: ``out_col`` and ``calibrated``.
    """
    required = {pred_col, actual_col, fold_col, horizon_col}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"calibrate_forecasts_rolling missing columns: {sorted(missing)}")

    df = predictions.copy()
    df[out_col] = np.nan
    df["calibrated"] = False

    def _fold_key(v):
        try:
            return (0, float(v))
        except (TypeError, ValueError):
            return (1, str(v))

    for h, h_df in df.groupby(horizon_col):
        folds_sorted = sorted(h_df[fold_col].dropna().unique(), key=_fold_key)
        for f in folds_sorted:
            cur_mask = (df[horizon_col] == h) & (df[fold_col] == f)
            prior_mask = (df[horizon_col] == h) & df[fold_col].apply(
                lambda x, f=f: _fold_key(x) < _fold_key(f)
            )
            prior = df.loc[prior_mask, [pred_col, actual_col]].dropna()
            if log_space:
                prior = prior[(prior[pred_col] > 0) & (prior[actual_col] > 0)]
            cur_pred = df.loc[cur_mask, pred_col]

            if len(prior) < min_prior_obs:
                df.loc[cur_mask, out_col] = cur_pred.values
                continue

            if log_space:
                x = np.log(prior[pred_col].values)
                y = np.log(prior[actual_col].values)
            else:
                x = prior[pred_col].values.astype(float)
                y = prior[actual_col].values.astype(float)
            a, b = _ols_ab(x, y)

            p = cur_pred.values.astype(float)
            if log_space:
                with np.errstate(divide="ignore", invalid="ignore"):
                    calib = np.exp(a + b * np.log(np.where(p > 0, p, np.nan)))
                calib = np.where(np.isfinite(calib), calib, p)
            else:
                calib = a + b * p
            df.loc[cur_mask, out_col] = calib
            df.loc[cur_mask, "calibrated"] = True

    return df


def reliability_table(
    predicted,
    actual,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Binned reliability table: mean predicted vs mean realised per quantile bin.

    A perfectly calibrated forecast has ``mean_actual == mean_predicted`` in every bin; a
    systematic gap between the two columns is level miscalibration. Bins are equal-count
    quantile bins of ``predicted``.

    Returns a DataFrame with ``bin``, ``n``, ``mean_predicted``, ``mean_actual``.
    """
    df = pd.DataFrame(
        {"predicted": np.asarray(predicted, dtype=float), "actual": np.asarray(actual, dtype=float)}
    ).dropna()
    if df.empty:
        return pd.DataFrame(columns=["bin", "n", "mean_predicted", "mean_actual"])
    n_bins = max(1, min(n_bins, df["predicted"].nunique()))
    df["bin"] = pd.qcut(df["predicted"].rank(method="first"), n_bins, labels=False)
    grouped = df.groupby("bin")
    return pd.DataFrame(
        {
            "bin": np.arange(n_bins),
            "n": grouped.size().reindex(range(n_bins)).fillna(0).astype(int).values,
            "mean_predicted": grouped["predicted"].mean().reindex(range(n_bins)).values,
            "mean_actual": grouped["actual"].mean().reindex(range(n_bins)).values,
        }
    )
