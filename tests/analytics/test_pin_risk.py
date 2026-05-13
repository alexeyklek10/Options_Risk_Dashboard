"""Pin-risk likelihood scores."""

from __future__ import annotations

from datetime import date

import pandas as pd

from ord.analytics.pin_risk import pin_risk


def test_pin_risk_top_candidate_is_oi_concentrated_atm(synthetic_chain: pd.DataFrame) -> None:
    near_expiry = date(2026, 6, 19)
    result = pin_risk(synthetic_chain, near_expiry)
    assert result is not None
    assert len(result.candidates) > 0
    # Fixture concentrates OI at K=100 (the ATM strike); top candidate should be 100.
    assert result.candidates[0].strike == 100.0
    assert result.candidates[0].score > 50.0


def test_pin_risk_empty_chain() -> None:
    assert pin_risk(pd.DataFrame(), date(2026, 6, 19)) is None


def test_pin_risk_returns_none_for_unknown_expiry() -> None:
    chain = pd.DataFrame(
        {
            "expiry": [date(2026, 6, 19)],
            "strike": [100.0],
            "option_type": ["call"],
            "open_interest": [100],
            "underlying_price": [100.0],
            "dte": [30],
            "implied_vol": [0.2],
        }
    )
    assert pin_risk(chain, date(2026, 12, 19)) is None
