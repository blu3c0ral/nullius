"""Uniform result type for every check in the library.

The output of ``nullius`` is not a float — it is an *audit document*. Every public
check returns a :class:`CheckResult`; a run of many checks is collected into a
:class:`Report` that renders to a findings-style Markdown document. This mirrors the
research culture the library was extracted from: a diagnostic that returns only a
number invites you to skim it; one that returns a verdict, a statistic, a threshold,
and the evidence rows forces you to read it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: Severity levels in increasing order of seriousness.
SEVERITIES = ("info", "warning", "fatal")

_SEVERITY_RANK = {"info": 0, "warning": 1, "fatal": 2}


@dataclass
class CheckResult:
    """The verdict of a single diagnostic check.

    Attributes
    ----------
    name:
        Stable identifier for the check, e.g. ``"prefix_invariance"``.
    passed:
        ``True`` if the check passed, ``False`` if it failed, ``None`` for a purely
        informational result that carries no pass/fail semantics.
    severity:
        One of ``"info"``, ``"warning"``, ``"fatal"`` — how much a failure matters.
    statistic:
        The scalar the verdict is based on (the DM stat, the PBO, the DSR, ...), or
        ``None`` when the check has no single summarising number.
    threshold:
        The value ``statistic`` was compared against to decide ``passed``.
    details:
        One paragraph of human explanation: *what* failed and *why it matters*.
    evidence:
        The rows that prove the verdict (e.g. the mismatched decisions), or ``None``.
    """

    name: str
    passed: bool | None
    severity: str
    statistic: float | None = None
    threshold: float | None = None
    details: str = ""
    evidence: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if self.severity not in _SEVERITY_RANK:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")

    @property
    def failed(self) -> bool:
        """``True`` only when the check explicitly failed (``passed is False``)."""
        return self.passed is False

    def _verdict_str(self) -> str:
        if self.passed is None:
            return "INFO"
        return "PASS" if self.passed else f"FAIL[{self.severity}]"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        parts = [f"{self.name}: {self._verdict_str()}"]
        if self.statistic is not None:
            stat = f"statistic={self.statistic:.6g}"
            if self.threshold is not None:
                stat += f" threshold={self.threshold:.6g}"
            parts.append(stat)
        if self.details:
            parts.append(self.details)
        return " | ".join(parts)


class Report:
    """An ordered collection of :class:`CheckResult` that renders as an audit doc."""

    def __init__(self, results: list[CheckResult] | None = None) -> None:
        self.results: list[CheckResult] = list(results or [])

    def add(self, result: CheckResult) -> None:
        """Append a check result to the report."""
        if not isinstance(result, CheckResult):
            raise TypeError(f"Report.add expects a CheckResult, got {type(result)!r}")
        self.results.append(result)

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def worst(self) -> str:
        """Return the worst *failing* severity: ``"fatal"``, ``"warning"``, or ``"clean"``.

        Only checks that actually failed (``passed is False``) count — a passing
        check that happens to carry ``severity="fatal"`` does not make the report
        fatal.
        """
        rank = 0
        for r in self.results:
            if r.failed:
                rank = max(rank, _SEVERITY_RANK[r.severity])
        return {0: "clean", 1: "warning", 2: "fatal"}[rank]

    def render_markdown(self) -> str:
        """Render the report as a findings-style Markdown audit document."""
        icon = {"clean": "✅", "warning": "⚠️", "fatal": "❌"}[self.worst()]
        lines = [
            f"# nullius audit — {icon} {self.worst().upper()}",
            "",
            f"{len(self.results)} check(s) run.",
            "",
            "| check | verdict | statistic | threshold |",
            "| --- | --- | --- | --- |",
        ]
        for r in self.results:
            stat = "" if r.statistic is None else f"{r.statistic:.6g}"
            thr = "" if r.threshold is None else f"{r.threshold:.6g}"
            lines.append(f"| {r.name} | {r._verdict_str()} | {stat} | {thr} |")
        for r in self.results:
            if not r.details and (r.evidence is None or r.evidence.empty):
                continue
            lines += ["", f"## {r.name} — {r._verdict_str()}"]
            if r.details:
                lines += ["", r.details]
            if r.evidence is not None and not r.evidence.empty:
                head = r.evidence.head(10)
                lines += [
                    "",
                    f"Evidence ({len(r.evidence)} row(s), showing up to 10):",
                    "",
                    "```",
                    head.to_string(index=False),
                    "```",
                ]
        return "\n".join(lines)
