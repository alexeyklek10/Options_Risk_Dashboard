"""Shared synthetic-chain fixtures for analytics tests.

The fixture is deterministic and small: one underlying, two expiries, a
strike grid spanning +/- 20 percent of spot, sensible bid/ask spreads, OI
distributed so max-pain has an unambiguous answer for the close-expiry, and
a smile-shaped IV pattern.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from ord.data.base import CHAIN_COLUMNS

_FETCHED_AT = datetime(2026, 5, 13, 14, 30, tzinfo=UTC)
_TODAY = date(2026, 5, 13)


def _make_row(
    expiry: date,
    strike: float,
    option_type: str,
    spot: float,
    iv: float,
    oi: int,
    volume: int = 100,
) -> dict[str, object]:
    intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    time_value = max(0.5, iv * spot * 0.1)
    mid = intrinsic + time_value
    return {
        "ticker": "TST",
        "expiry": expiry,
        "dte": (expiry - _TODAY).days,
        "strike": strike,
        "option_type": option_type,
        "bid": mid - 0.05,
        "ask": mid + 0.05,
        "mid": mid,
        "last": mid,
        "volume": volume,
        "open_interest": oi,
        "implied_vol": iv,
        "underlying_price": spot,
        "fetched_at": _FETCHED_AT,
        "source": "yfinance",
    }


@pytest.fixture()
def synthetic_chain() -> pd.DataFrame:
    """Two-expiry chain centered on spot=100 with a v-shaped smile."""
    spot = 100.0
    near_expiry = date(2026, 6, 19)  # 37 DTE
    far_expiry = date(2026, 9, 18)  # 128 DTE
    strikes = np.arange(80.0, 121.0, 5.0)
    rows: list[dict[str, object]] = []
    for expiry in (near_expiry, far_expiry):
        for strike in strikes:
            # Smile: highest IV at the wings, lowest at ATM.
            moneyness = abs(strike - spot) / spot
            iv = 0.18 + 0.4 * moneyness
            # Near-expiry: max OI at K=100 -> max-pain should pin near 100.
            # Far-expiry: roughly uniform OI for testing aggregation.
            oi = (5000 if strike == 100.0 else 500) if expiry == near_expiry else 200
            for option_type in ("call", "put"):
                rows.append(_make_row(expiry, float(strike), option_type, spot, iv, oi))
    return pd.DataFrame(rows, columns=CHAIN_COLUMNS)
