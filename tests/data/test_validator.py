"""CrossSourceValidator: divergence metrics across provider chains."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from ord.data.base import CHAIN_COLUMNS
from ord.data.validator import cross_validate


def _row(provider: str, iv: float, mid: float, oi: int) -> dict[str, object]:
    return {
        "ticker": "TST",
        "expiry": date(2026, 6, 19),
        "dte": 30,
        "strike": 100.0,
        "option_type": "call",
        "bid": mid - 0.05,
        "ask": mid + 0.05,
        "mid": mid,
        "last": mid,
        "volume": 100,
        "open_interest": oi,
        "implied_vol": iv,
        "underlying_price": 100.0,
        "fetched_at": datetime(2026, 5, 13, 14, 30, tzinfo=timezone.utc),
        "source": provider,
    }


def test_cross_validate_handles_empty_chains() -> None:
    result = cross_validate({})
    assert result.per_row.empty
    assert result.pair_aggregates.empty
    assert result.iv_calibration.empty


def test_cross_validate_records_iv_range_and_pair_aggregates() -> None:
    yf_df = pd.DataFrame([_row("yfinance", 0.20, 5.0, 1000)], columns=CHAIN_COLUMNS)
    tr_df = pd.DataFrame([_row("tradier", 0.22, 5.1, 950)], columns=CHAIN_COLUMNS)
    result = cross_validate({"yfinance": yf_df, "tradier": tr_df})
    assert len(result.per_row) == 1
    row = result.per_row.iloc[0]
    assert abs(row["iv_range"] - 0.02) < 1e-9
    assert row["n_providers"] == 2
    assert not result.pair_aggregates.empty
    pair_row = result.pair_aggregates.iloc[0]
    assert pair_row["pair"] == "tradier vs yfinance"
    assert pair_row["n_overlap"] == 1


def test_cross_validate_skips_singleton_contracts() -> None:
    yf_df = pd.DataFrame([_row("yfinance", 0.20, 5.0, 1000)], columns=CHAIN_COLUMNS)
    only_one = cross_validate({"yfinance": yf_df})
    assert only_one.per_row.empty


def test_calibration_residual_is_populated_when_recomputed_iv_succeeds() -> None:
    yf_df = pd.DataFrame([_row("yfinance", 0.20, 5.0, 1000)], columns=CHAIN_COLUMNS)
    tr_df = pd.DataFrame([_row("tradier", 0.21, 5.05, 900)], columns=CHAIN_COLUMNS)
    result = cross_validate({"yfinance": yf_df, "tradier": tr_df})
    assert not result.iv_calibration.empty
    assert "abs_residual" in result.iv_calibration.columns
