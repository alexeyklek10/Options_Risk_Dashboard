"""Max-pain: pinning strike + dollar-pain magnitude."""

from __future__ import annotations

from datetime import date

import pandas as pd

from ord.analytics.max_pain import max_pain_all_expiries, max_pain_for_expiry


def test_max_pain_pins_near_concentration(synthetic_chain: pd.DataFrame) -> None:
    near_expiry = date(2026, 6, 19)
    result = max_pain_for_expiry(synthetic_chain, near_expiry)
    assert result is not None
    # OI concentrated at K=100 in the fixture; max-pain should land there.
    assert result.strike == 100.0
    assert result.dollar_pain >= 0


def test_max_pain_returns_none_when_no_oi() -> None:
    empty_chain = pd.DataFrame(
        {
            "expiry": [date(2026, 6, 19)] * 4,
            "strike": [95.0, 100.0, 105.0, 110.0],
            "option_type": ["call", "call", "put", "put"],
            "open_interest": [0, 0, 0, 0],
        }
    )
    assert max_pain_for_expiry(empty_chain, date(2026, 6, 19)) is None


def test_max_pain_curve_is_full_strike_grid(synthetic_chain: pd.DataFrame) -> None:
    result = max_pain_for_expiry(synthetic_chain, date(2026, 6, 19))
    assert result is not None
    assert len(result.pain_curve) > 0
    # Pain curve is monotonically structured: min at the pin strike.
    pin_pain = result.pain_curve.loc[result.pain_curve["strike"] == result.strike, "pain"].iloc[0]
    assert pin_pain == result.pain_curve["pain"].min()


def test_max_pain_all_expiries_returns_one_per_expiry(synthetic_chain: pd.DataFrame) -> None:
    results = max_pain_all_expiries(synthetic_chain)
    assert set(results.keys()) == set(synthetic_chain["expiry"].unique())
