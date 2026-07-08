"""Example 3 — the overfitting tax on a grid search.

We run ~200 moving-average-crossover configurations over a synthetic random-walk price
(pure noise). The single best in-sample Sharpe looks impressive — but:

* PBO (Probability of Backtest Overfitting) is elevated: the in-sample winner is unreliable
  out-of-sample.
* the Deflated Sharpe of the winner, given ~200 trials, is not significant (< 0.95).

Then we plant one genuine edge and rerun: PBO collapses toward zero and the DSR of the
(now real) winner clears 0.95. Same machinery, opposite verdict.

Run: ``python examples/03_pbo_of_a_gridsearch.py``
"""

import numpy as np

import nullius as nl


def gridsearch_returns(rng: np.random.Generator, plant_edge: bool) -> np.ndarray:
    """(T x N) per-period returns for N MA-crossover configs on a synthetic price."""
    T = 1500
    price = 100 * np.cumprod(1 + rng.normal(0, 0.01, T))
    fast_windows = range(3, 23)
    slow_windows = range(30, 130, 10)

    cols = []
    for fast in fast_windows:
        for slow in slow_windows:
            ma_fast = _sma(price, fast)
            ma_slow = _sma(price, slow)
            position = np.where(ma_fast > ma_slow, 1.0, -1.0)
            ret = position[:-1] * (price[1:] / price[:-1] - 1.0)
            cols.append(ret)
    M = np.column_stack(cols)
    if plant_edge:
        M[:, 0] += 0.0015  # one config gets a genuine, stable positive drift
    return M


def _sma(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    csum = np.cumsum(x)
    out[w - 1 :] = (csum[w - 1 :] - np.concatenate([[0], csum[:-w]])) / w
    out[: w - 1] = x[: w - 1]
    return out


def report(label: str, M: np.ndarray) -> None:
    n_trials = M.shape[1]
    pbo = nl.pbo(M, n_splits=14, seed=0)
    best = int(np.argmax(M.mean(axis=0) / M.std(axis=0, ddof=1)))
    per_period_sr = M.mean(axis=0) / M.std(axis=0, ddof=1)
    V = nl.trial_sharpe_variance(per_period_sr, annualized=False)
    dsr = nl.deflated_sharpe(M[:, best], n_trials=n_trials, var_across_trials=V)
    print(f"== {label} ({n_trials} configs) ==")
    print(f"  PBO={pbo.statistic:.3f}  -> {pbo._verdict_str()}")
    print(f"  best-config DSR={dsr.statistic:.3f}  -> {dsr._verdict_str()}\n")


def main() -> None:
    rng = np.random.default_rng(7)
    report("PURE NOISE", gridsearch_returns(rng, plant_edge=False))
    report("ONE PLANTED EDGE", gridsearch_returns(rng, plant_edge=True))


if __name__ == "__main__":
    main()
