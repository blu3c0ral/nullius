# Roadmap

`nullius` **v0.1.0** ships the general, frequency-blind core — leakage, placebo,
multiplicity, metrics, forecast evaluation, and robustness — covering the five backtest
failure modes (leak, luck, selection, fragile inference, regime concentration) at any
fixed-bar frequency.

Two optional extras are planned as **future directions**. Both are deferred; each needs a
short design pass before implementation (the open questions are listed so the work can be
picked up cleanly).

---

## `nullius[forensics]` — target-validation harness *(planned)*

**What it is.** For machine-learning *forecasting* models, an autopsy of whether the
model's apparent predictive skill is real. Five controlled "sabotage" tests — if a score
barely drops when the input is sabotaged, the skill was that artefact:

1. **Lag-only baseline** — predict from past values of the target only; the headline
   verdict is `skill_ratio = full_feature_IC / lag_only_IC` (`< 1` = the model is just
   riding the target's autocorrelation, not the features).
2. **De-overlap** — keep every *k*-th row so multi-bar labels no longer overlap; a score
   that collapses was inflated by non-independent observations.
3. **Drift-strip (demean)** — remove the slow-moving mean; a score that vanishes was only
   capturing drift.
4. **Alternate horizons** — recompute the target at neighbouring horizons; a real signal
   survives, a one-window fluke does not.
5. **Feature shuffle** — give one asset another's features; a score that survives means the
   model is scoring off something other than a real feature→outcome relationship.

**Open design questions to resolve first.**
- Rebuild a **generic per-fold metric layer** (IC / hit-rate / EV and their aggregation) on
  top of a user-supplied `fit_predict(X_train, y_train, X_test) -> np.ndarray` — the metric
  surface currently lives inside a private training loop, not in a reusable form.
- Return a **`Report` of per-test `CheckResult`s** carrying the skill-ratio verdict and a
  threshold (match the core convention; the original returns nested dicts).
- The **alternate-horizons** test needs target *recomputation*: accept a
  `recompute_targets(data, horizon)` callable and skip the test when it is absent.
- Pin the `(X, y, group, time)` panel contract and the fold source (reuse `purged_splits`);
  express all windows in **bars**.

**Why later.** Narrow audience (ML-forecasting quants) and the most implementation effort —
it is a redesign rather than a straight port, since the harness must be decoupled from a
specific training stack. This is the one extra that *drives* a model (via `fit_predict`)
rather than only consuming its outputs.

---

## `nullius[intraday]` — high-frequency gate *(planned)*

**What it is.** The failure modes that are specific to intraday / high-frequency data —
they cannot occur on daily bars, or the daily defaults become actively wrong when you go
fast. Ships behind an optional extra so a daily user never sees the added complexity; an
intraday user is forced through it. Split into a high-value leakage half (M4a) and a
noise-robust-estimator half (M4b).

**M4a — intraday leakage layer.**
- `check_execution_timing` (static) and `one_switch_leakage` (behavioural) — the latter runs
  a strategy twice, once "cheating" (decide and fill on the same bar) and once honestly
  (fill at the next bar's open), and measures the Sharpe the cheating manufactured. The
  intraday twin of `check_prefix_invariance`.
- Session-aware splits and overlapping-label concurrency (average uniqueness, sequential
  bootstrap).

**M4b — noise-robust estimators.**
- Seasonality-preserving placebos (time-of-day blocks; a Fourier deseasonalizer is
  research-grade).
- Microstructure-noise-robust volatility (two-scale realized volatility + a signature-plot
  warning; the realized kernel is research-grade).
- Autocorrelation-aware Sharpe annualisation (Lo 2002) and event bars (tick / volume /
  dollar), whose returns are closer to independent and make every downstream statistic more
  valid.

**Open design questions to resolve first.**
- `one_switch_leakage` needs **its own pipeline contract** (accepts an execution-convention
  toggle and emits *returns*) — distinct from the core's `Callable[[DataFrame], DataFrame]`.
- `check_execution_timing` needs a **decision/signal timestamp** on trades, a **bars OHLC
  contract**, and defined **latency units**.
- Build a **fatal-polarity `CheckResult`** for the leaky-vs-clean Sharpe gap (do not reuse
  `sharpe_difference_test`'s info/significance polarity; inner-join before diffing so the
  two runs' unequal lengths do not raise).
- **Re-specify** session-aware purging around *label maturation across the session gap* plus
  a `labels`/`spans` frame — the shipped bar-count splitter already skips overnight gaps, so
  the original motivation needs restating.
- Lo-2002 annualisation is **data-dependent**: compute the factor from the return series;
  the `Clock` only carries the *flag*. Reclassify the exact Patton-Politis-White block-length
  constant, the realized kernel, and the Fourier deseasonalizer as research-grade
  (build-from-paper) and **non-gating** for the release.
- Add the `[intraday]` / `[forensics]` extras to `pyproject.toml` and wire the
  `MissingExtraError` import gate (the pattern already exists on the `Clock`).

**Out of scope even here.** Order-book / queue-position simulation, latency-arbitrage
modelling, and tick-data cleaning. `nullius` audits the *statistics* of a backtest; it is
not an execution simulator.

**Sequencing.** M4a (execution-timing + sessions) is the highest-value, most-buildable
increment; M4b (estimators) follows.
