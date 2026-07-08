"""Example 2 — a real signal vs a persistence-only signal, judged against AR(1) placebos.

Two synthetic predictors of next-period returns:

* ``skilled``  = 0.1 * ret_{t+1} + noise  (genuinely peeks at a slice of the future return)
* ``persist``  = a purely autocorrelated series with no relationship to returns

We score each by its information coefficient, then locate that score inside a distribution
of AR(1)-matched placebos (noise with the same persistence). The skilled signal clears the
placebo floor; the persistence-only one does not — which is the whole point: beating AR(1)
noise rejects "no skill", and a persistent-but-useless series fails to.

Run: ``python examples/02_placebo_vs_real_signal.py``
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import nullius as nl


def ic(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(spearmanr(pred, actual).statistic)


def main() -> None:
    rng = np.random.default_rng(1)
    n = 500
    future_ret = rng.normal(0, 1, n)

    skilled = 0.1 * future_ret + rng.normal(0, 1, n)
    persist = np.empty(n)
    persist[0] = rng.normal()
    for i in range(1, n):
        persist[i] = 0.85 * persist[i - 1] + rng.normal(0, np.sqrt(1 - 0.85**2))

    frame = pd.DataFrame({"asset": "X", "predicted": skilled})
    persist_frame = pd.DataFrame({"asset": "X", "predicted": persist})

    real_ic_skilled = ic(skilled, future_ret)
    real_ic_persist = ic(persist, future_ret)

    # AR(1)-matched placebos of each series; score their IC against the same future returns
    placebo_ic_skilled = [
        ic(nl.ar1_matched_placebo(frame, seed=s)["predicted"].to_numpy(), future_ret)
        for s in range(50)
    ]
    placebo_ic_persist = [
        ic(nl.ar1_matched_placebo(persist_frame, seed=s)["predicted"].to_numpy(), future_ret)
        for s in range(50)
    ]

    skilled_cmp = nl.compare_against_placebos(abs(real_ic_skilled), np.abs(placebo_ic_skilled))
    persist_cmp = nl.compare_against_placebos(abs(real_ic_persist), np.abs(placebo_ic_persist))

    print(f"SKILLED  IC={real_ic_skilled:+.3f}  -> {skilled_cmp.details}\n")
    print(f"PERSIST  IC={real_ic_persist:+.3f}  -> {persist_cmp.details}\n")

    assert skilled_cmp.passed is True
    assert persist_cmp.passed is False


if __name__ == "__main__":
    main()
