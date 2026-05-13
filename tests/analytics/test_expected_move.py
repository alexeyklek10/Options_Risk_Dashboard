"""Expected-move = 0.85 * ATM-straddle-mid."""

from __future__ import annotations

import pandas as pd
import pytest

from ord.analytics.expected_move import DEFAULT_SCALING, expected_move


def test_expected_move_uses_atm_straddle(synthetic_chain: pd.DataFrame) -> None:
    result = expected_move(synthetic_chain)
    assert len(result.per_expiry) == synthetic_chain["expiry"].nunique()
    row = result.per_expiry.iloc[0]
    assert row["atm_strike"] == pytest.approx(100.0)
    assert row["expected_move"] == pytest.approx(DEFAULT_SCALING * row["straddle_mid"])


def test_expected_move_custom_scaling(synthetic_chain: pd.DataFrame) -> None:
    result = expected_move(synthetic_chain, scaling=1.0)
    assert result.scaling == 1.0
    row = result.per_expiry.iloc[0]
    assert row["expected_move"] == pytest.approx(row["straddle_mid"])


def test_expected_move_missing_atm_mid_yields_nan(synthetic_chain: pd.DataFrame) -> None:
    chain = synthetic_chain.copy()
    # Wipe the ATM call mid for one expiry.
    near = chain["expiry"].min()
    mask = (chain["expiry"] == near) & (chain["strike"] == 100.0) & (chain["option_type"] == "call")
    chain.loc[mask, "mid"] = None
    result = expected_move(chain)
    affected = result.per_expiry[result.per_expiry["expiry"] == near].iloc[0]
    assert pd.isna(affected["expected_move"])


def test_expected_move_empty_chain() -> None:
    empty = pd.DataFrame()
    result = expected_move(empty)
    assert result.per_expiry.empty


def test_expected_move_missing_put_side_skipped() -> None:
    only_calls = pd.DataFrame(
        [
            {
                "expiry": "2026-06-19",
                "strike": 100.0,
                "option_type": "call",
                "mid": 5.0,
                "underlying_price": 100.0,
            }
        ]
    )
    result = expected_move(only_calls)
    assert pd.isna(result.per_expiry.iloc[0]["expected_move"])
