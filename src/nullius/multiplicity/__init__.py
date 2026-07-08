"""Multiple-testing deflation: honest trial counting, Deflated Sharpe, and PBO/CSCV."""

from __future__ import annotations

from .dsr import (
    deflated_sharpe,
    deflated_sharpe_ratio,
    trial_sharpe_variance,
)
from .pbo import pbo, probability_of_backtest_overfitting
from .trials import TrialRegistry

__all__ = [
    "TrialRegistry",
    "deflated_sharpe",
    "deflated_sharpe_ratio",
    "trial_sharpe_variance",
    "pbo",
    "probability_of_backtest_overfitting",
]
