"""Example 1 — catch a planted lookahead in under a minute.

We build a random-walk prediction panel (pure noise: there is no signal), then gate on it
two ways:

* a LEAKY gate that thresholds on the *full-sample* quantile, and
* a CLEAN gate that thresholds on an *expanding* quantile (past data only).

``check_prefix_invariance`` re-runs each pipeline on truncated inputs and proves whether
any decision dated <= T changed. The leaky gate is flagged fatal; the clean one passes.

Run: ``python examples/01_planted_lookahead.py``
"""

import numpy as np
import pandas as pd

import nullius as nl


def make_panel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2020-01-01", periods=250, freq="B")
    frames = []
    for asset in ("SPY", "QQQ", "IWM"):
        # a random walk — deliberately no predictive content
        predicted = np.cumsum(rng.normal(0, 1, len(times)))
        frames.append(pd.DataFrame({"time": times, "asset": asset, "predicted": predicted}))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    data = make_panel()

    leaky = nl.check_prefix_invariance(nl.leaky_global_quantile_gate, data)
    clean = nl.check_prefix_invariance(nl.clean_expanding_quantile_gate, data)

    print("LEAKY full-sample-quantile gate:")
    print(
        f"  passed={leaky.passed}  severity={leaky.severity}  mismatched_rows={leaky.statistic:.0f}"
    )
    print(f"  {leaky.details}\n")
    if leaky.evidence is not None:
        print("  evidence (first rows):")
        print(leaky.evidence.head(5).to_string(index=False), "\n")

    print("CLEAN expanding-quantile gate:")
    print(f"  passed={clean.passed}  (no lookahead detected)\n")

    report = nl.Report([leaky, clean])
    print(f"Overall verdict: {report.worst().upper()}")

    assert leaky.passed is False, "the leaky gate should be flagged"
    assert clean.passed is True, "the clean gate should pass"


if __name__ == "__main__":
    main()
