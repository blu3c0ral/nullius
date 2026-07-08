"""Example 4 — honest volatility-forecast evaluation on a synthetic GARCH(1,1) series.

We simulate a GARCH(1,1) return process whose conditional variance we know exactly, then
compare two forecasts of that variance:

* ``true``  = the actual conditional variance (the oracle), and
* ``stale`` = the conditional variance from 20 periods ago (a badly lagged forecast).

We evaluate with QLIKE loss, a Mincer-Zarnowitz calibration test, and a Diebold-Mariano
test. QLIKE and DM both prefer the true model; MZ flags the stale forecast as
miscalibrated (beta far from 1).

Run: ``python examples/04_forecast_eval_synthetic.py``
"""

import numpy as np
import pandas as pd

import nullius as nl


def simulate_garch(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    omega, alpha, beta = 1e-6, 0.08, 0.90
    var = np.empty(n)
    ret = np.empty(n)
    var[0] = omega / (1 - alpha - beta)
    for t in range(n):
        ret[t] = np.sqrt(var[t]) * rng.standard_normal()
        if t + 1 < n:
            var[t + 1] = omega + alpha * ret[t] ** 2 + beta * var[t]
    return ret, var


def main() -> None:
    rng = np.random.default_rng(3)
    ret, cond_var = simulate_garch(rng, 1500)
    realised_var = ret**2  # noisy but unbiased proxy for the conditional variance

    lag = 20
    true_forecast = cond_var
    stale_forecast = np.concatenate([np.full(lag, cond_var[0]), cond_var[:-lag]])

    q_true = nl.qlike_mean(realised_var, true_forecast)
    q_stale = nl.qlike_mean(realised_var, stale_forecast)
    print(f"QLIKE  true={q_true:.4g}   stale={q_stale:.4g}   (lower is better)\n")

    mz = nl.mincer_zarnowitz(realised_var, stale_forecast, horizon=1)
    print(
        f"Mincer-Zarnowitz on STALE forecast: beta={mz['beta']:.3f}, "
        f"Wald p={mz['wald_p_value']:.3g} (H0: alpha=0, beta=1)\n"
    )

    losses = pd.DataFrame(
        {
            "time": pd.date_range("2018-01-01", periods=len(ret), freq="B"),
            "loss_diff": nl.qlike_loss(realised_var, true_forecast).to_numpy()
            - nl.qlike_loss(realised_var, stale_forecast).to_numpy(),
        }
    )
    dm = nl.dm_test(losses)
    print(f"Diebold-Mariano (true vs stale): {dm.details}")

    assert q_true < q_stale


if __name__ == "__main__":
    main()
