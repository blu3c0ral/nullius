<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo-dark.svg">
    <img src="docs/logo-light.svg" alt="Nullius" width="88" height="88">
  </picture>
</p>

<h1 align="center">Nullius</h1>

<p align="center">
  A falsify-first toolkit for backtests: lookahead detection with negative controls,<br>
  skill placebos, honest forecast evaluation, and multiple-testing deflation.
</p>

---

Every backtesting library — backtrader, vectorbt, zipline, bt — helps you **produce** a
performance number. Almost nothing helps you **attack** one. `nullius` is the missing
"falsify-first" layer: the methodological weapons professionals use to prove a result is
real — purged cross-validation, prefix-invariance leak detection, AR(1)-matched placebos,
Deflated Sharpe, Probability of Backtest Overfitting — packaged, tested, and documented so
a retail quant can `pip install` them and answer: *is this Sharpe 1.2 real, or did I
manufacture it?*

## The war story

A real lookahead bug once survived four weeks inside a *careful* research process. A single
global statistic — a full-sample quantile — was computed once and used inside a
per-timestamp decision. It looked innocent. It inflated a Calmar ratio by ~30%. What
finally killed it was not a sharper eye; it was a **regression test with a negative
control** — a test that first proved it could *detect* the leak by firing on a deliberately
broken pipeline, then proved the fix by staying silent on the corrected one.

That is the philosophy this library is built on, stated verbatim in its tests:

> **A leak detector you have never seen fire is itself untested.**

Every probe in `nullius` ships with a negative control that must fire in CI.

## 30-second quickstart

```python
import numpy as np, pandas as pd
import nullius as nl

# a pure-noise prediction panel — there is no signal here
t = pd.date_range("2020-01-01", periods=250, freq="B")
data = pd.DataFrame({"time": np.repeat(t, 3),
                     "asset": ["SPY", "QQQ", "IWM"] * len(t),
                     "predicted": np.random.default_rng(0).normal(size=3 * len(t))})

# leaky_global_quantile_gate thresholds on the FULL-SAMPLE quantile — a lookahead
leak = nl.check_prefix_invariance(nl.leaky_global_quantile_gate, data)
print(leak.passed, leak.severity)     # False fatal  — decisions dated <= T changed
print(leak.evidence.head())           # the exact rows that moved

# the expanding-quantile version uses only past data and passes
ok = nl.check_prefix_invariance(nl.clean_expanding_quantile_gate, data)
print(ok.passed)                      # True
```

## What this is / is not

- **It is** a set of *checks*. Each returns a `CheckResult` — a verdict, a statistic, a
  threshold, and the evidence rows — and many together render as a Markdown audit document.
  The output is not a float; it is a document you are meant to *read*.
- **It is not** a backtester. It does not produce signals, place trades, or fetch data. It
  audits the *statistics* of a backtest you already have. All examples run on synthetic data
  generated in-repo.

## The five failure modes

| Failure mode | Question it answers | Module |
|---|---|---|
| **Leak** | Did a decision use future data? | `leakage` — `check_prefix_invariance`, `purged_splits` |
| **Luck** | Is the "edge" better than a skill-free null? | `placebo` — `ar1_matched_placebo`, `compare_against_placebos` |
| **Selection** | Did I overfit by trying many configs? | `multiplicity` — `deflated_sharpe`, `pbo`, `TrialRegistry` |
| **Fragile inference** | Is my ΔSharpe / forecast edge significant? | `metrics`, `forecast_eval` — `sharpe_difference_test`, `dm_test`, `mincer_zarnowitz` |
| **Regime concentration** | Does it survive out of one lucky regime? | `robustness` — `subperiod_ic_stability`, `regime_stratified_metrics` |

## Any timeframe

The general core is **frequency-blind**: it operates on observation counts, not the
calendar, so DSR, PBO, prefix-invariance, QLIKE/MZ/DM, and the equity metrics are correct at
any fixed-bar frequency. Every function that needs time-scaling takes an optional
`clock: Clock`; the `DAILY` preset supplies the 252 / √252 defaults, and any other frequency
is one `Clock(periods_per_year=...)` away. The frequency-*sensitive* checks (session-aware
purging, intraday seasonal placebos, microstructure-noise-robust estimators) are planned for
the optional `nullius[intraday]` extra.

## Install

```bash
pip install nullius                 # core: numpy, pandas, scipy
pip install "nullius[reference]"    # + statsmodels, only for cross-check tests
```

Requires Python ≥ 3.11. Core runtime depends only on numpy / pandas / scipy — Mincer-Zarnowitz
and rolling calibration are implemented in numpy (their HAC covariance reproduces
`statsmodels` to ~1e-12; the optional `[reference]` extra runs that cross-check).

## Examples

Runnable, synthetic-data-only, each under 100 lines:

- [`examples/01_planted_lookahead.py`](examples/01_planted_lookahead.py) — catch a planted lookahead.
- [`examples/02_placebo_vs_real_signal.py`](examples/02_placebo_vs_real_signal.py) — a real signal vs a persistence-only one, judged against AR(1) placebos.
- [`examples/03_pbo_of_a_gridsearch.py`](examples/03_pbo_of_a_gridsearch.py) — the overfitting tax on a 200-config grid search.
- [`examples/04_forecast_eval_synthetic.py`](examples/04_forecast_eval_synthetic.py) — QLIKE + Mincer-Zarnowitz + Diebold-Mariano on a synthetic GARCH series.

## Citations

Patton (2011, *J. Econometrics*) — QLIKE robust loss · Diebold & Mariano (1995) + Harvey,
Leybourne & Newbold (1997) — equal-predictive-accuracy test + small-sample correction ·
Mincer & Zarnowitz (1969) — forecast efficiency · Bailey & López de Prado (2014, *JPM*) —
Deflated Sharpe · Bailey, Borwein, López de Prado & Zhu (2017) — Probability of Backtest
Overfitting (CSCV) · Jobson & Korkie (1981) + Memmel (2003) — Sharpe-difference test · Lo
(2002) — Sharpe standard error · Politis & Romano (1994) — stationary bootstrap · López de
Prado (2018, *Advances in Financial ML*) — purging & embargo.

## License

MIT.
