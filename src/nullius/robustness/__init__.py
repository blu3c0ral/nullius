"""Robustness: bootstrap IC significance and regime/subperiod stratification."""

from __future__ import annotations

from .bootstrap import fisher_combined_pvalue, ic_bootstrap_pvalue
from .subperiod import (
    US_EQUITY_REGIMES,
    regime_stratified_metrics,
    subperiod_ic_stability,
)

__all__ = [
    "ic_bootstrap_pvalue",
    "fisher_combined_pvalue",
    "subperiod_ic_stability",
    "regime_stratified_metrics",
    "US_EQUITY_REGIMES",
]
