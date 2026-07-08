"""TrialRegistry — make N (the number of trials searched) honest and mechanical.

With LLM-assisted iteration, untracked trials silently multiply: every tweak-and-rerun
is another draw from the multiple-testing lottery, but only the winners get written down.
An append-only registry records *every* invocation so the trial count fed into the
Deflated Sharpe haircut reflects what actually happened, not what you remember.

I/O policy: the registry is the one component allowed to touch disk, and only through an
explicitly injected ``path``. There is no hidden wall clock — ``timestamp`` is always a
caller-supplied value, so a logged run is fully reproducible and replayable.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd


class TrialRegistry:
    """Append-only log of backtest trials, optionally persisted to a JSONL file.

    Parameters
    ----------
    path:
        Optional path to a JSONL file. If given and it exists, prior trials are loaded;
        every :meth:`log` appends one line. If ``None``, the registry is in-memory only.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._trials: list[dict[str, Any]] = []
        if self.path is not None and self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self._trials.append(json.loads(line))

    def log(
        self,
        name: str,
        params: Mapping[str, Any],
        metrics: Mapping[str, Any],
        timestamp: Any,
    ) -> None:
        """Record one trial. ``timestamp`` is caller-supplied (no hidden clock)."""
        record = {
            "name": str(name),
            "params": dict(params),
            "metrics": dict(metrics),
            "timestamp": timestamp,
        }
        self._trials.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")

    def n_trials(self) -> int:
        """Number of trials logged so far — the honest N for a multiplicity correction."""
        return len(self._trials)

    def __len__(self) -> int:
        return len(self._trials)

    def __iter__(self):
        return iter(self._trials)

    def metric_values(self, key: str) -> list[float]:
        """All finite values of ``metrics[key]`` across logged trials."""
        out = []
        for t in self._trials:
            v = t.get("metrics", {}).get(key)
            if v is not None:
                out.append(float(v))
        return out

    def to_frame(self) -> pd.DataFrame:
        """Flatten the registry to a DataFrame with ``params.*`` / ``metrics.*`` columns."""
        rows = []
        for t in self._trials:
            row: dict[str, Any] = {"name": t["name"], "timestamp": t["timestamp"]}
            for k, v in t.get("params", {}).items():
                row[f"params.{k}"] = v
            for k, v in t.get("metrics", {}).items():
                row[f"metrics.{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def tracked(
        self,
        fn: Callable | None = None,
        *,
        name: str | None = None,
    ) -> Callable:
        """Decorator that logs every call of a backtest function as a trial.

        The wrapped function's keyword arguments become ``params`` and its return value
        (a mapping, else ``{"value": result}``) becomes ``metrics``. The ``timestamp`` is
        the trial ordinal at log time — a deterministic, reproducible sequence id rather
        than a wall clock. Use as ``@registry.tracked`` or ``@registry.tracked(name=...)``.
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                metrics = dict(result) if isinstance(result, Mapping) else {"value": result}
                self.log(name or func.__name__, dict(kwargs), metrics, self.n_trials())
                return result

            return wrapper

        if fn is not None and callable(fn):
            return decorator(fn)
        return decorator
