"""Golden-parity tests: every ported function must reproduce the source repo's output.

Fixtures were frozen once from ``strategies.vol_spike.diagnostics`` (see the extraction
script). The source repo is NOT a test dependency here — only the frozen JSON is read.
Exact ports must match within 1e-12; the two OLS-based ports (calibration, MZ) are numpy
reimplementations validated against the statsmodels source within a tight relative
tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import nullius as nl

EXACT = {"rtol": 0, "atol": 1e-12}


def test_dsr_parity(golden):
    d = golden("dsr")
    out = nl.deflated_sharpe_ratio(np.array(d["returns"]), **d["params"])
    for k, v in d["output"].items():
        if isinstance(v, (int, float)) and v == v:  # skip NaN/bool-as-int carefully
            assert abs(out[k] - v) < 1e-12, (k, out[k], v)


def test_pbo_parity(golden):
    p = golden("pbo")
    out = nl.probability_of_backtest_overfitting(np.array(p["matrix"]), **p["params"])
    assert abs(out["pbo"] - p["output"]["pbo"]) < 1e-12
    assert abs(out["median_logit"] - p["output"]["median_logit"]) < 1e-12
    np.testing.assert_allclose(out["lambda_dist"], p["output"]["lambda_dist"], **EXACT)


def test_trial_sharpe_variance_parity(golden):
    t = golden("trial_sharpe_variance")
    a = nl.trial_sharpe_variance(t["sharpes"], annualized=True, periods_per_year=252)
    pp = nl.trial_sharpe_variance(t["sharpes"], annualized=False)
    assert abs(a - t["output_annualized"]) < 1e-15
    assert abs(pp - t["output_per_period"]) < 1e-15


def test_placebo_parity(golden):
    pl = golden("placebo")
    pin = pd.DataFrame(pl["input"])
    ar1 = nl.ar1_matched_placebo(pin, seed=pl["seed"])
    uni = nl.random_uniform_placebo(pin, seed=pl["seed"])
    np.testing.assert_allclose(ar1["predicted"].to_numpy(), pl["ar1_predicted"], **EXACT)
    np.testing.assert_allclose(uni["predicted"].to_numpy(), pl["uniform_predicted"], **EXACT)


def test_equity_parity(golden):
    e = golden("equity")
    prices = pd.DataFrame(e["prices"])
    prices["time"] = pd.to_datetime(prices["time"])
    trades = pd.DataFrame(e["trades"])
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])

    dl = nl.build_equity_curve(trades, prices, e["universe"], cost_per_side_bps=6.0)
    port = dl.drop_duplicates("time").set_index("time")["portfolio_ret"]
    np.testing.assert_allclose(port.to_numpy(), e["portfolio_ret"]["value"], **EXACT)

    m = nl.equity_metrics(port)
    for k in ["cagr", "vol_ann", "sharpe", "sortino", "mdd", "calmar", "total_return"]:
        assert abs(m[k] - float(e["metrics"][k])) < 1e-12, k

    bh = nl.buy_and_hold_curve(prices, e["universe"])
    np.testing.assert_allclose(bh["bh_ret"].to_numpy(), e["bh"]["bh_ret"], **EXACT)
    np.testing.assert_allclose(bh["bh_equity"].to_numpy(), e["bh"]["bh_equity"], **EXACT)

    ex = nl.exposure_stats(
        trades, e["universe"], test_start=e["test_start"], test_end=e["test_end"]
    )
    np.testing.assert_array_equal(ex["n_trades"].to_numpy(), e["exposure"]["n_trades"])
    np.testing.assert_array_equal(ex["total_hold_periods"].to_numpy(), e["exposure"]["total_hold"])
    np.testing.assert_allclose(
        ex["pct_periods_in_market"].to_numpy(), e["exposure"]["pct_in_market"], **EXACT
    )


def test_qlike_parity(golden):
    fe = golden("forecast_eval")
    ql = nl.qlike_loss(np.array(fe["actual_var"]), np.array(fe["pred1"])).to_numpy()
    np.testing.assert_allclose(ql, fe["qlike_loss"], **EXACT)


def test_dm_parity(golden):
    fe = golden("forecast_eval")
    dm = nl.diebold_mariano(
        np.array(fe["actual_var"]),
        np.array(fe["pred1"]),
        np.array(fe["pred2"]),
        loss="qlike",
        horizon=fe["horizon"],
    )
    for k in [
        "dm_stat",
        "dm_p_value",
        "dm_stat_hln",
        "dm_p_value_hln",
        "mean_diff",
        "loss1_mean",
        "loss2_mean",
    ]:
        assert abs(dm[k] - fe["dm_output"][k]) < 1e-12, k
    assert dm["better_model"] == fe["dm_output"]["better_model"]


def test_mz_parity_vs_statsmodels_source(golden):
    fe = golden("forecast_eval")
    mz = nl.mincer_zarnowitz(
        np.array(fe["actual_var"]), np.array(fe["pred1"]), horizon=fe["horizon"]
    )
    exp = fe["mz_output"]
    # numpy Newey-West HAC reproduces the statsmodels source within 1e-9 relative
    for k in ["alpha", "beta", "alpha_se", "beta_se", "wald_stat", "r_squared"]:
        rel = abs(mz[k] - exp[k]) / (abs(exp[k]) + 1e-12)
        assert rel < 1e-9, (k, mz[k], exp[k], rel)


def test_calibration_parity(golden):
    c = golden("calibration")
    co = nl.calibrate_forecasts_rolling(pd.DataFrame(c["input"]), min_prior_obs=250, log_space=True)
    np.testing.assert_allclose(
        co["predicted_calibrated"].to_numpy(), c["predicted_calibrated"], rtol=1e-9, atol=1e-12
    )
    np.testing.assert_array_equal(co["calibrated"].to_numpy(), c["calibrated"])


def test_bootstrap_parity(golden):
    b = golden("bootstrap")
    boot = nl.ic_bootstrap_pvalue(
        pd.Series(b["predictions"]), pd.Series(b["actuals"]), **b["params"]
    )
    for k, v in b["ic_output"].items():
        assert abs(boot[k] - v) < 1e-12, k
    pvals = pd.Series([np.nan if x is None else x for x in b["fisher_pvalues"]])
    fish = nl.fisher_combined_pvalue(pvals)
    assert abs(fish["chi2"] - b["fisher_output"]["chi2"]) < 1e-12
    assert abs(fish["combined_p"] - b["fisher_output"]["combined_p"]) < 1e-12


def test_subperiod_parity(golden):
    s = golden("subperiod")
    sin = pd.DataFrame(s["input"])
    sin["time"] = pd.to_datetime(sin["time"])
    so = nl.subperiod_ic_stability(sin, s["periods"], per_asset=True)
    got = so.sort_values(["period", "asset"]).reset_index(drop=True)
    exp = sorted(s["output"], key=lambda r: (r["period"], r["ticker"]))
    assert len(got) == len(exp)
    for (_, row), er in zip(got.iterrows(), exp, strict=True):
        assert abs(row["ic"] - er["ic"]) < 1e-12
        assert row["n_obs"] == er["n_obs"]
