"""Equity-curve metrics and Sharpe inference."""

from __future__ import annotations

from .equity import (
    build_equity_curve,
    buy_and_hold_curve,
    compare_to_benchmark,
    equity_metrics,
    exposure_stats,
)
from .inference import sharpe_difference_test, sharpe_se

__all__ = [
    "build_equity_curve",
    "equity_metrics",
    "buy_and_hold_curve",
    "exposure_stats",
    "compare_to_benchmark",
    "sharpe_se",
    "sharpe_difference_test",
]
