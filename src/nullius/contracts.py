"""Data contracts — the three canonical frames the library operates on.

Every public entry point takes one of these frames and nothing else: no disk paths,
no universe lists, no strategy-specific defaults. The time key is always ``time``
(``datetime64``, which already carries sub-second resolution), and the asset key is
always ``asset``. Validators raise :class:`ContractError` naming the offending column,
so a misnamed frame fails loudly at the boundary rather than silently deep inside a
computation.

+-------------+--------------------------------------------------------------+
| Contract    | Required columns                                             |
+=============+==============================================================+
| predictions | ``time`` (datetime64), ``asset`` (str), ``predicted`` (float)|
|             | ``actual`` (float) is optional.                              |
+-------------+--------------------------------------------------------------+
| trades      | ``asset``, ``entry_time``, ``exit_time``, ``net_return``     |
|             | (fractional per-trade). ``entry_price``/``exit_price``/      |
|             | ``weight`` are optional.                                     |
+-------------+--------------------------------------------------------------+
| returns     | ``time``, ``ret`` (float) in long form; or ``time`` plus one |
|             | column per asset in wide form.                               |
+-------------+--------------------------------------------------------------+
"""

from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt


class ContractError(ValueError):
    """Raised when a frame violates one of the data contracts."""


def _require_columns(df: pd.DataFrame, required: set[str], contract: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ContractError(f"{contract} must be a pandas DataFrame, got {type(df)!r}")
    missing = required - set(df.columns)
    if missing:
        raise ContractError(
            f"{contract} frame is missing required column(s) {sorted(missing)}; "
            f"present columns are {sorted(df.columns)}"
        )


def _require_datetime(df: pd.DataFrame, col: str, contract: str) -> None:
    if not pdt.is_datetime64_any_dtype(df[col]):
        raise ContractError(
            f"{contract} column {col!r} must be datetime64 "
            f"(got dtype {df[col].dtype}); convert with pandas.to_datetime first"
        )


def _require_numeric(df: pd.DataFrame, col: str, contract: str) -> None:
    if not pdt.is_numeric_dtype(df[col]):
        raise ContractError(
            f"{contract} column {col!r} must be numeric (got dtype {df[col].dtype})"
        )


def validate_predictions(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    asset_col: str = "asset",
    pred_col: str = "predicted",
    require_actual: bool = False,
) -> None:
    """Validate a *predictions* frame in place, raising :class:`ContractError`.

    Detects the failure mode of feeding a mislabeled or wrong-dtype frame into a
    check — which otherwise surfaces as a confusing error deep inside numpy — by
    naming the exact missing or non-conforming column at the boundary.
    """
    required = {time_col, asset_col, pred_col}
    if require_actual:
        required.add("actual")
    _require_columns(df, required, "predictions")
    _require_datetime(df, time_col, "predictions")
    _require_numeric(df, pred_col, "predictions")
    if "actual" in df.columns:
        _require_numeric(df, "actual", "predictions")


def validate_trades(
    df: pd.DataFrame,
    *,
    asset_col: str = "asset",
    entry_col: str = "entry_time",
    exit_col: str = "exit_time",
    ret_col: str = "net_return",
) -> None:
    """Validate a *trades* frame in place, raising :class:`ContractError`.

    Guards against silently treating a frame with the wrong time or return columns
    as a valid trade log, which would produce a plausible-looking but meaningless
    equity curve.
    """
    _require_columns(df, {asset_col, entry_col, exit_col, ret_col}, "trades")
    _require_datetime(df, entry_col, "trades")
    _require_datetime(df, exit_col, "trades")
    _require_numeric(df, ret_col, "trades")


def validate_returns(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    ret_col: str = "ret",
) -> None:
    """Validate a *returns* frame (long or wide) in place, raising :class:`ContractError`.

    A long frame must carry ``time`` and ``ret``; a wide frame must carry ``time``
    and at least one numeric asset column. Detects passing an unindexed or
    all-non-numeric frame where a return series was expected.
    """
    _require_columns(df, {time_col}, "returns")
    _require_datetime(df, time_col, "returns")
    if ret_col in df.columns:
        _require_numeric(df, ret_col, "returns")
        return
    value_cols = [c for c in df.columns if c != time_col]
    numeric_cols = [c for c in value_cols if pdt.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ContractError(
            "returns frame must contain either a numeric 'ret' column (long form) "
            "or at least one numeric per-asset column (wide form); "
            f"found no numeric value columns among {value_cols}"
        )
