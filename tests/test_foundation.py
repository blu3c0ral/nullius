"""Tests for the M0 foundation: result, contracts, clock."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nullius import (
    DAILY,
    CheckResult,
    Clock,
    ContractError,
    MissingExtraError,
    Report,
    Session,
    validate_predictions,
    validate_returns,
    validate_trades,
)


class TestCheckResult:
    def test_rejects_bad_severity(self):
        with pytest.raises(ValueError, match="severity"):
            CheckResult(name="x", passed=True, severity="critical")

    def test_failed_property(self):
        assert CheckResult("x", False, "fatal").failed is True
        assert CheckResult("x", True, "fatal").failed is False
        assert CheckResult("x", None, "info").failed is False  # informational

    def test_str_contains_verdict(self):
        r = CheckResult("prefix", False, "fatal", statistic=3.0, threshold=0.0, details="leak")
        assert "FAIL[fatal]" in str(r)
        assert "leak" in str(r)


class TestReport:
    def test_worst_only_counts_failures(self):
        rep = Report()
        rep.add(CheckResult("a", True, "fatal"))  # passing fatal-severity check
        rep.add(CheckResult("b", False, "warning"))
        assert rep.worst() == "warning"

    def test_worst_fatal(self):
        rep = Report([CheckResult("a", False, "fatal"), CheckResult("b", False, "warning")])
        assert rep.worst() == "fatal"

    def test_worst_clean(self):
        rep = Report([CheckResult("a", True, "fatal"), CheckResult("b", None, "info")])
        assert rep.worst() == "clean"

    def test_render_markdown_has_table_and_evidence(self):
        ev = pd.DataFrame({"time": [1, 2], "issue": ["x", "y"]})
        rep = Report([CheckResult("prefix", False, "fatal", 2.0, 0.0, "boom", ev)])
        md = rep.render_markdown()
        assert "nullius audit" in md
        assert "prefix" in md
        assert "Evidence" in md

    def test_add_type_checked(self):
        with pytest.raises(TypeError):
            Report().add("not a check result")


class TestContracts:
    def _preds(self):
        return pd.DataFrame(
            {
                "time": pd.to_datetime(["2021-01-01", "2021-01-02"]),
                "asset": ["A", "A"],
                "predicted": [0.1, 0.2],
            }
        )

    def test_predictions_ok(self):
        validate_predictions(self._preds())

    def test_predictions_missing_column(self):
        df = self._preds().drop(columns=["predicted"])
        with pytest.raises(ContractError, match="predicted"):
            validate_predictions(df)

    def test_predictions_requires_datetime(self):
        df = self._preds()
        df["time"] = ["2021-01-01", "2021-01-02"]  # strings
        with pytest.raises(ContractError, match="datetime64"):
            validate_predictions(df)

    def test_predictions_require_actual(self):
        with pytest.raises(ContractError, match="actual"):
            validate_predictions(self._preds(), require_actual=True)

    def test_trades_ok_and_bad(self):
        tr = pd.DataFrame(
            {
                "asset": ["A"],
                "entry_time": pd.to_datetime(["2021-01-01"]),
                "exit_time": pd.to_datetime(["2021-01-05"]),
                "net_return": [0.02],
            }
        )
        validate_trades(tr)
        with pytest.raises(ContractError, match="net_return"):
            validate_trades(tr.drop(columns=["net_return"]))

    def test_returns_long_and_wide(self):
        long = pd.DataFrame({"time": pd.to_datetime(["2021-01-01"]), "ret": [0.01]})
        validate_returns(long)
        wide = pd.DataFrame({"time": pd.to_datetime(["2021-01-01"]), "A": [0.01], "B": [0.02]})
        validate_returns(wide)

    def test_returns_no_numeric_columns(self):
        df = pd.DataFrame({"time": pd.to_datetime(["2021-01-01"]), "label": ["x"]})
        with pytest.raises(ContractError, match="numeric"):
            validate_returns(df)

    def test_not_a_dataframe(self):
        with pytest.raises(ContractError, match="DataFrame"):
            validate_predictions([1, 2, 3])


class TestClock:
    def test_daily_defaults(self):
        assert DAILY.periods_per_year == 252
        assert DAILY.sharpe_factor() == pytest.approx(np.sqrt(252))

    def test_intraday_factor(self):
        c = Clock(periods_per_year=252 * 78)
        assert c.sharpe_factor() == pytest.approx(np.sqrt(252 * 78))

    def test_lo2002_needs_extra(self):
        c = Clock(periods_per_year=252, ann_method="lo2002")
        with pytest.raises(MissingExtraError, match="intraday"):
            c.sharpe_factor()

    def test_validation(self):
        with pytest.raises(ValueError, match="periods_per_year"):
            Clock(periods_per_year=0)
        with pytest.raises(ValueError, match="bar_type"):
            Clock(periods_per_year=252, bar_type="hourly")
        with pytest.raises(ValueError, match="ann_method"):
            Clock(periods_per_year=252, ann_method="bad")

    def test_session_carried(self):
        c = Clock(periods_per_year=252 * 78, session=Session("09:30", "16:00"))
        assert c.session.open == "09:30"
