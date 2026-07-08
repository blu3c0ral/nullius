"""Frequency generality — the ``Clock`` object is the gate between the general core
and the intraday extra.

The library is organised in three tiers:

1. **General core (frequency-blind).** DSR, PBO/CSCV, prefix-invariance, QLIKE/MZ/DM,
   equity/Sharpe/Calmar, placebo z-scores. These operate on *observation counts* (T),
   not the calendar, and are correct at any fixed-bar frequency once every horizon and
   window is expressed in bars rather than calendar days.
2. **The ``DAILY`` preset.** The only genuinely daily-flavoured configuration: it
   supplies ``periods_per_year=252`` and √-time Sharpe annualisation. It is a
   *configuration* of the general core, not a separate implementation.
3. **The ``nullius[intraday]`` extra.** The frequency-*sensitive* checks (session-aware
   purging, intraday seasonal placebos, microstructure-noise-robust estimators). These
   live behind an optional dependency and ship the ``INTRADAY`` preset.

Every function that needs a notion of time scaling takes an optional ``clock: Clock``;
if omitted it defaults to ``DAILY``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Annualisation methods understood by the core. ``"lo2002"`` (autocorrelation-adjusted)
#: is supplied by the ``nullius[intraday]`` extra; the core validates but does not
#: implement it.
_ANN_METHODS = ("sqrt", "lo2002")

_BAR_TYPES = ("time", "tick", "volume", "dollar", "imbalance")


class MissingExtraError(ImportError):
    """Raised when code from an optional extra is used without the extra installed."""


@dataclass(frozen=True)
class Session:
    """A trading session, used only by the intraday extra for session-aware purging.

    Carried on the ``Clock`` so daily code can ignore it entirely. ``open`` and
    ``close`` are ``"HH:MM"`` strings in timezone ``tz``.
    """

    open: str
    close: str
    tz: str = "America/New_York"
    overnight_gap: bool = True


@dataclass(frozen=True)
class Clock:
    """Time-scaling configuration shared across the library.

    Parameters
    ----------
    periods_per_year:
        Observations per year — 252 for daily bars; ``252 * 6.5 * 12`` for 5-minute
        regular-trading-hours equity bars; and so on. Drives every annualisation.
    bar_type:
        Which clock the pipeline runs on: ``"time"`` (fixed calendar bars) or one of
        the event-driven bar types from the intraday extra.
    session:
        Optional :class:`Session`; only the intraday extra consults it.
    hac_lag:
        Fixed HAC/Newey-West truncation lag, or ``None`` to let each check pick a
        data-driven default.
    ann_method:
        ``"sqrt"`` for √-time annualisation (correct for iid returns) or ``"lo2002"``
        for the autocorrelation-adjusted variant supplied by the intraday extra.
    """

    periods_per_year: float
    bar_type: str = "time"
    session: Session | None = None
    hac_lag: int | None = None
    ann_method: str = "sqrt"

    def __post_init__(self) -> None:
        if self.periods_per_year <= 0:
            raise ValueError(f"periods_per_year must be positive, got {self.periods_per_year}")
        if self.bar_type not in _BAR_TYPES:
            raise ValueError(f"bar_type must be one of {_BAR_TYPES}, got {self.bar_type!r}")
        if self.ann_method not in _ANN_METHODS:
            raise ValueError(f"ann_method must be one of {_ANN_METHODS}, got {self.ann_method!r}")
        if self.hac_lag is not None and self.hac_lag < 0:
            raise ValueError(f"hac_lag must be non-negative or None, got {self.hac_lag}")

    def sharpe_factor(self) -> float:
        """Multiplicative factor to annualise a per-period Sharpe ratio.

        For ``ann_method="sqrt"`` this is ``sqrt(periods_per_year)``. The
        autocorrelation-adjusted ``"lo2002"`` factor is provided by the intraday
        extra and raises :class:`MissingExtraError` here.
        """
        if self.ann_method == "sqrt":
            return float(np.sqrt(self.periods_per_year))
        raise MissingExtraError(
            "ann_method='lo2002' requires the intraday extra: pip install nullius[intraday]"
        )


#: The default preset for daily bars — the only genuinely daily-flavoured config.
DAILY = Clock(periods_per_year=252.0)
