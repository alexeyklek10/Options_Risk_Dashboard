"""Skew analytics: ATM IV interpolation, 25-delta strikes, RR/BF, near-ATM slope."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ord.analytics.skew import skew


def test_skew_returns_one_row_per_expiry(synthetic_chain: pd.DataFrame) -> None:
    result = skew(synthetic_chain)
    assert len(result.per_expiry) == synthetic_chain["expiry"].nunique()


def test_atm_iv_is_smile_minimum(synthetic_chain: pd.DataFrame) -> None:
    # The fixture's v-smile has its minimum at K=spot=100; ATM IV should be
    # smaller than any wing IV.
    result = skew(synthetic_chain)
    row = result.per_expiry.iloc[0]
    wing_ivs = synthetic_chain[synthetic_chain["strike"].isin([80.0, 120.0])][
        "implied_vol"
    ].dropna()
    assert float(row["atm_iv"]) < float(wing_ivs.min())


def test_butterfly_positive_for_smile(synthetic_chain: pd.DataFrame) -> None:
    # A genuine smile (wings higher than ATM) should give bf_25d > 0.
    result = skew(synthetic_chain)
    bf = result.per_expiry["bf_25d"].dropna()
    assert (bf > 0).any()


def test_risk_reversal_symmetric_smile_near_zero(synthetic_chain: pd.DataFrame) -> None:
    # Symmetric v-smile -> RR close to zero.
    result = skew(synthetic_chain)
    rr = result.per_expiry["rr_25d"].dropna()
    assert np.allclose(rr.to_numpy(dtype=np.float64), 0.0, atol=0.05)


def test_slope_close_to_zero_for_symmetric_smile(synthetic_chain: pd.DataFrame) -> None:
    # Symmetric smile -> slope near zero (slight, due to interpolation choices).
    result = skew(synthetic_chain)
    slope = result.per_expiry["slope"].dropna()
    assert (slope.abs() < 0.5).all()


def test_skew_empty_chain() -> None:
    empty = pd.DataFrame()
    result = skew(empty)
    assert result.per_expiry.empty


def test_skew_handles_chain_with_no_provider_iv() -> None:
    chain = pd.DataFrame(
        {
            "expiry": ["2026-06-19"] * 4,
            "dte": [37] * 4,
            "strike": [95.0, 100.0, 105.0, 110.0],
            "option_type": ["call", "call", "put", "put"],
            "implied_vol": [None, None, None, None],
            "underlying_price": [100.0] * 4,
        }
    )
    result = skew(chain)
    assert result.per_expiry.empty
