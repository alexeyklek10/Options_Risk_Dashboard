"""Breeden-Litzenberger implied PDF."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from ord.analytics.implied_pdf import implied_pdf_all_expiries, implied_pdf_for_expiry
from ord.pricing.black_scholes import bs_price


def _bs_call_chain_for_expiry(
    spot: float, sigma: float, T: float, r: float, expiry: date
) -> pd.DataFrame:
    strikes = np.linspace(spot * 0.7, spot * 1.3, 25)
    rows = []
    for k in strikes:
        price = bs_price(spot, float(k), T, r, sigma, "call", 0.0)
        rows.append(
            {
                "expiry": expiry,
                "dte": int(T * 365),
                "strike": float(k),
                "option_type": "call",
                "mid": price,
                "underlying_price": spot,
            }
        )
    return pd.DataFrame(rows)


def test_implied_pdf_recovers_lognormal_for_bs_prices() -> None:
    spot = 100.0
    sigma = 0.20
    T = 0.5
    r = 0.03
    expiry = date(2026, 11, 13)
    chain = _bs_call_chain_for_expiry(spot, sigma, T, r, expiry)
    result = implied_pdf_for_expiry(chain, expiry, r=r, smoothing=0.0)
    assert result is not None
    # Density should integrate to ~1.
    integral = float(trapezoid(result.density, result.strikes))
    assert abs(integral - 1.0) < 0.02
    # Mode should land near the forward, which for BS is spot * exp((r-q)*T).
    forward = spot * math.exp(r * T)
    mode_strike = float(result.strikes[int(np.argmax(result.density))])
    assert abs(mode_strike - forward) < spot * 0.10


def test_implied_pdf_returns_none_when_insufficient_quotes() -> None:
    chain = pd.DataFrame(
        {
            "expiry": [date(2026, 6, 19)] * 3,
            "dte": [37, 37, 37],
            "strike": [95.0, 100.0, 105.0],
            "option_type": ["call", "call", "call"],
            "mid": [6.0, 3.0, 1.0],
            "underlying_price": [100.0, 100.0, 100.0],
        }
    )
    assert implied_pdf_for_expiry(chain, date(2026, 6, 19)) is None


def test_implied_pdf_all_expiries_returns_dict() -> None:
    spot = 100.0
    sigma = 0.25
    r = 0.02
    chain = pd.concat(
        [
            _bs_call_chain_for_expiry(spot, sigma, 0.25, r, date(2026, 8, 14)),
            _bs_call_chain_for_expiry(spot, sigma, 0.5, r, date(2026, 11, 13)),
        ]
    )
    results = implied_pdf_all_expiries(chain, r=r)
    assert set(results.keys()) == {date(2026, 8, 14), date(2026, 11, 13)}
