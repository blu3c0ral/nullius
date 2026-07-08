"""nullius — falsify-first backtest diagnostics.

Named after *nullius in verba* ("take nobody's word for it"). Every backtesting library
helps you *produce* a performance number; ``nullius`` helps you *attack* one — lookahead
detection with negative controls, skill placebos, honest forecast evaluation, and
multiple-testing deflation.

The public surface is re-exported here; submodules
(:mod:`nullius.leakage`, :mod:`nullius.placebo`, :mod:`nullius.multiplicity`,
:mod:`nullius.metrics`, :mod:`nullius.forecast_eval`, :mod:`nullius.robustness`) carry the
full detail.
"""

from __future__ import annotations

from .clock import DAILY, Clock, MissingExtraError, Session
from .contracts import (
    ContractError,
    validate_predictions,
    validate_returns,
    validate_trades,
)
from .forecast_eval import (
    calibrate_forecasts_rolling,
    diebold_mariano,
    diebold_mariano_panel,
    dm_stat,
    dm_test,
    mincer_zarnowitz,
    qlike_loss,
    qlike_mean,
    reliability_table,
)
from .leakage import (
    Split,
    check_prefix_invariance,
    check_split_purging,
    clean_expanding_quantile_gate,
    leaky_global_quantile_gate,
    leaky_unpurged_splits,
    purged_splits,
)
from .metrics import (
    build_equity_curve,
    buy_and_hold_curve,
    compare_to_benchmark,
    equity_metrics,
    exposure_stats,
    sharpe_difference_test,
    sharpe_se,
)
from .multiplicity import (
    TrialRegistry,
    deflated_sharpe,
    deflated_sharpe_ratio,
    pbo,
    probability_of_backtest_overfitting,
    trial_sharpe_variance,
)
from .placebo import (
    ar1_matched_placebo,
    compare_against_placebos,
    random_uniform_placebo,
    shuffled_entries_placebo,
)
from .result import CheckResult, Report
from .robustness import (
    US_EQUITY_REGIMES,
    fisher_combined_pvalue,
    ic_bootstrap_pvalue,
    regime_stratified_metrics,
    subperiod_ic_stability,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # foundation
    "Clock",
    "DAILY",
    "Session",
    "MissingExtraError",
    "CheckResult",
    "Report",
    "ContractError",
    "validate_predictions",
    "validate_trades",
    "validate_returns",
    # leakage
    "check_prefix_invariance",
    "purged_splits",
    "check_split_purging",
    "Split",
    "leaky_global_quantile_gate",
    "clean_expanding_quantile_gate",
    "leaky_unpurged_splits",
    # placebo
    "ar1_matched_placebo",
    "random_uniform_placebo",
    "shuffled_entries_placebo",
    "compare_against_placebos",
    # multiplicity
    "TrialRegistry",
    "deflated_sharpe",
    "deflated_sharpe_ratio",
    "trial_sharpe_variance",
    "pbo",
    "probability_of_backtest_overfitting",
    # metrics
    "build_equity_curve",
    "equity_metrics",
    "buy_and_hold_curve",
    "exposure_stats",
    "compare_to_benchmark",
    "sharpe_se",
    "sharpe_difference_test",
    # forecast eval
    "qlike_loss",
    "qlike_mean",
    "mincer_zarnowitz",
    "dm_stat",
    "dm_test",
    "diebold_mariano",
    "diebold_mariano_panel",
    "calibrate_forecasts_rolling",
    "reliability_table",
    # robustness
    "ic_bootstrap_pvalue",
    "fisher_combined_pvalue",
    "subperiod_ic_stability",
    "regime_stratified_metrics",
    "US_EQUITY_REGIMES",
]
