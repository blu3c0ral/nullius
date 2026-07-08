# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-07-07

First functional release (`0.0.0` was the name-reservation placeholder): the general
(frequency-blind) core, milestones M1 + M2 shipped together.

### Added — M1 (MVP)

- **Foundation:** `Clock`/`Session`/`DAILY` (frequency gate), `CheckResult`/`Report`
  (audit-document result type), data contracts (`validate_predictions`/`trades`/`returns`).
- **Leakage** (`nullius.leakage`): `check_prefix_invariance` (flagship lookahead probe),
  `purged_splits` / `check_split_purging`, and the negative controls
  `leaky_global_quantile_gate`, `clean_expanding_quantile_gate`, `leaky_unpurged_splits`.
- **Placebo** (`nullius.placebo`): `ar1_matched_placebo`, `random_uniform_placebo`,
  `shuffled_entries_placebo`, and `compare_against_placebos`.
- **Multiplicity** (`nullius.multiplicity`): `TrialRegistry`, `deflated_sharpe_ratio` /
  `deflated_sharpe`, `probability_of_backtest_overfitting` / `pbo`, `trial_sharpe_variance`.
- **Metrics** (`nullius.metrics`): `build_equity_curve`, `equity_metrics`,
  `buy_and_hold_curve`, `exposure_stats`, `compare_to_benchmark`, `sharpe_se`,
  `sharpe_difference_test`.
- Examples 1–3; property tests (DSR, PBO, prefix-invariance); negative-control-fires tests;
  golden-parity fixtures frozen from the source repo.

### Added — M2

- **Forecast evaluation** (`nullius.forecast_eval`): `qlike_loss` / `qlike_mean`,
  `mincer_zarnowitz`, the layered Diebold-Mariano API (`dm_stat` kernel, `diebold_mariano`,
  `diebold_mariano_panel`, `dm_test` dispatcher with the time-clustered panel variant),
  `calibrate_forecasts_rolling`, `reliability_table`.
- **Robustness** (`nullius.robustness`): `ic_bootstrap_pvalue`, `fisher_combined_pvalue`,
  `subperiod_ic_stability`, `regime_stratified_metrics`, `US_EQUITY_REGIMES`.
- Example 4; Sharpe-difference size/power property tests; statsmodels cross-check for MZ.

### Notes

- **`statsmodels` is not a runtime dependency.** Mincer-Zarnowitz and rolling calibration
  are implemented in numpy/scipy; `statsmodels` is used only by the optional cross-check
  test suite (`pip install nullius[reference]`). The numpy Newey-West HAC reproduces
  `statsmodels`' `use_correction=True` covariance to ~1e-12.
- Every ported function is verified byte-for-byte (1e-12) against the source repo via frozen
  golden fixtures; the source repo is not a runtime or test dependency.

### Not yet included

- M3 (`nullius.forensics`, the 5-test target-validation harness) and M4
  (`nullius[intraday]`: execution-timing, session-aware splits, seasonal placebos,
  microstructure estimators, event bars) are planned optional extras.
