"""IV vs realized vol spread.

Compares the ATM implied vol of the nearest expiry with at least 21 DTE
to the 21-day close-to-close annualized realized vol of the underlying.

Realized vol over a window of N daily log returns is::

    rv = sqrt(sum(r_i^2) / N) * sqrt(252)

where 252 is the annualization factor for US trading days.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ord.analytics.skew import skew

TRADING_DAYS_PER_YEAR: int = 252


@dataclass(frozen=True)
class IVRVResult:
    """IV vs realized vol snapshot."""

    realized_vol_21d: float | None
    atm_iv_21d: float | None
    spread: float | None
    expiry_used: date | None


def realized_vol(prices: pd.Series, window: int = 21) -> float | None:
    """Annualized close-to-close realized vol over the most recent ``window`` days."""
    if len(prices) < window + 1:
        return None
    returns = np.log(prices).diff().dropna()
    if len(returns) < window:
        return None
    sample = returns.tail(window)
    return float(np.sqrt((sample**2).sum() / window) * np.sqrt(TRADING_DAYS_PER_YEAR))


def iv_rv_spread(
    chain: pd.DataFrame,
    underlying_history: pd.Series,
    r: float = 0.04,
    q: float = 0.0,
    window: int = 21,
) -> IVRVResult:
    """ATM IV (nearest expiry past `window` DTE) minus realized vol over `window` days.

    Returns the spread (IV - RV); positive means IV is trading rich.
    """
    rv = realized_vol(underlying_history, window=window)

    sk = skew(chain, r=r, q=q)
    if sk.per_expiry.empty:
        return IVRVResult(
            realized_vol_21d=rv,
            atm_iv_21d=None,
            spread=None,
            expiry_used=None,
        )
    per = sk.per_expiry.copy()
    dtes = chain.groupby("expiry")["dte"].first()
    per["dte"] = per["expiry"].map(dtes)
    eligible = per[per["dte"] >= window].dropna(subset=["atm_iv"])
    if eligible.empty:
        return IVRVResult(
            realized_vol_21d=rv,
            atm_iv_21d=None,
            spread=None,
            expiry_used=None,
        )
    row = eligible.iloc[0]
    iv = float(row["atm_iv"])
    expiry = row["expiry"]
    expiry_d = expiry.date() if hasattr(expiry, "date") else expiry
    spread = iv - rv if (rv is not None) else None
    return IVRVResult(
        realized_vol_21d=rv,
        atm_iv_21d=iv,
        spread=spread,
        expiry_used=expiry_d,
    )
