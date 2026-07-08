"""Leakage detection — the differentiator.

- :func:`check_prefix_invariance` (flagship): re-run a pipeline on truncated input and
  prove no output dated ``<= T`` changed.
- :func:`purged_splits` / :func:`check_split_purging`: build and audit walk-forward folds
  that purge the label-overlap gap between train and test.
- Negative controls (:func:`leaky_global_quantile_gate`, :func:`clean_expanding_quantile_gate`,
  :func:`leaky_unpurged_splits`): reference bad/good pipelines the probes are tested against.
"""

from __future__ import annotations

from .negative_controls import (
    clean_expanding_quantile_gate,
    leaky_global_quantile_gate,
    leaky_unpurged_splits,
)
from .prefix import check_prefix_invariance
from .splits import Split, check_split_purging, purged_splits

__all__ = [
    "check_prefix_invariance",
    "purged_splits",
    "check_split_purging",
    "Split",
    "leaky_global_quantile_gate",
    "clean_expanding_quantile_gate",
    "leaky_unpurged_splits",
]
