"""Put-call ratio aggregates + per-expiry + rolling helper."""

from __future__ import annotations

import pandas as pd
import pytest

from ord.analytics.put_call_ratio import put_call_ratio, rolling_pcr


def test_pcr_returns_floats_when_both_sides_present(synthetic_chain: pd.DataFrame) -> None:
    result = put_call_ratio(synthetic_chain)
    # Fixture has equal call/put OI and volume at every strike, so PCR ~= 1.
    assert result.by_volume_aggregate == pytest.approx(1.0, abs=1e-6)
    assert result.by_oi_aggregate == pytest.approx(1.0, abs=1e-6)


def test_pcr_per_expiry_has_one_row_per_expiry(synthetic_chain: pd.DataFrame) -> None:
    result = put_call_ratio(synthetic_chain)
    assert len(result.per_expiry) == synthetic_chain["expiry"].nunique()


def test_pcr_returns_none_for_zero_call_side() -> None:
    only_puts = pd.DataFrame(
        {
            "expiry": ["2026-06-19"],
            "strike": [100.0],
            "option_type": ["put"],
            "volume": [10],
            "open_interest": [100],
        }
    )
    result = put_call_ratio(only_puts)
    assert result.by_volume_aggregate is None
    assert result.by_oi_aggregate is None


def test_pcr_empty_chain() -> None:
    empty = pd.DataFrame()
    result = put_call_ratio(empty)
    assert result.by_volume_aggregate is None
    assert result.by_oi_aggregate is None
    assert result.per_expiry.empty


def test_rolling_pcr_returns_empty_when_below_window() -> None:
    history = pd.DataFrame({"pcr": [0.8, 0.9]}, index=pd.date_range("2026-01-01", periods=2))
    out = rolling_pcr(history, window=20)
    assert out.empty


def test_rolling_pcr_computes_rolling_mean() -> None:
    history = pd.DataFrame({"pcr": [0.5] * 25}, index=pd.date_range("2026-01-01", periods=25))
    out = rolling_pcr(history, window=20)
    assert not out.empty
    assert out.iloc[-1] == pytest.approx(0.5)


def test_rolling_pcr_handles_missing_column() -> None:
    history = pd.DataFrame({"other": [1.0] * 30}, index=pd.date_range("2026-01-01", periods=30))
    out = rolling_pcr(history)
    assert out.empty
