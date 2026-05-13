"""Canonical chain schema + DataProvider ABC fan-out default."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import ClassVar

import pandas as pd
import pytest
from pydantic import ValidationError

from ord.data.base import (
    CHAIN_COLUMNS,
    ChainRow,
    DataProvider,
    ProviderName,
    ProviderRateLimitError,
    _empty_chain,
)


def _row_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "ticker": "SPY",
        "expiry": date(2026, 6, 19),
        "dte": 37,
        "strike": 600.0,
        "option_type": "call",
        "bid": 5.10,
        "ask": 5.20,
        "mid": 5.15,
        "last": 5.18,
        "volume": 1234,
        "open_interest": 45678,
        "implied_vol": 0.18,
        "underlying_price": 595.42,
        "fetched_at": datetime(2026, 5, 13, 14, 30, tzinfo=timezone.utc),
        "source": "yfinance",
    }
    base.update(overrides)
    return base


def test_chain_row_accepts_minimum_valid_payload() -> None:
    row = ChainRow(**_row_kwargs())  # type: ignore[arg-type]
    assert row.ticker == "SPY"
    assert row.option_type == "call"


def test_chain_row_rejects_unknown_option_type() -> None:
    with pytest.raises(ValidationError):
        ChainRow(**_row_kwargs(option_type="straddle"))  # type: ignore[arg-type]


def test_chain_row_rejects_zero_strike() -> None:
    with pytest.raises(ValidationError):
        ChainRow(**_row_kwargs(strike=0.0))  # type: ignore[arg-type]


def test_chain_row_rejects_negative_dte() -> None:
    with pytest.raises(ValidationError):
        ChainRow(**_row_kwargs(dte=-1))  # type: ignore[arg-type]


def test_chain_row_allows_null_quotes() -> None:
    row = ChainRow(
        **_row_kwargs(  # type: ignore[arg-type]
            bid=None,
            ask=None,
            mid=None,
            last=None,
            volume=None,
            open_interest=None,
            implied_vol=None,
        )
    )
    assert row.bid is None and row.implied_vol is None


def test_chain_row_is_frozen() -> None:
    row = ChainRow(**_row_kwargs())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        row.ticker = "AAPL"  # type: ignore[misc]


def test_empty_chain_has_canonical_columns() -> None:
    df = _empty_chain()
    assert list(df.columns) == CHAIN_COLUMNS
    assert df.empty


# ---------------------------------------------------------------------------
# DataProvider ABC default get_full_chain
# ---------------------------------------------------------------------------


class _FakeProvider(DataProvider):
    """Minimal in-memory provider for ABC fan-out tests."""

    name: ClassVar[ProviderName] = "yfinance"

    def __init__(self, chains_by_expiry: dict[date, pd.DataFrame], price: float = 100.0) -> None:
        self._chains = chains_by_expiry
        self._price = price

    def get_underlying_price(self, ticker: str) -> float:  # noqa: ARG002
        return self._price

    def get_expirations(self, ticker: str) -> list[date]:  # noqa: ARG002
        return sorted(self._chains.keys())

    def get_chain(self, ticker: str, expiry: date) -> pd.DataFrame:  # noqa: ARG002
        chain = self._chains.get(expiry)
        if chain is None:
            return _empty_chain()
        if "raise" in chain.columns:
            raise RuntimeError("synthetic failure")
        return chain


def _toy_row(expiry: date, strike: float, option_type: str = "call") -> dict[str, object]:
    return {
        "ticker": "TST",
        "expiry": expiry,
        "dte": 30,
        "strike": strike,
        "option_type": option_type,
        "bid": 1.0,
        "ask": 1.1,
        "mid": 1.05,
        "last": 1.05,
        "volume": 10,
        "open_interest": 100,
        "implied_vol": 0.2,
        "underlying_price": 100.0,
        "fetched_at": datetime(2026, 5, 13, 14, 30, tzinfo=timezone.utc),
        "source": "yfinance",
    }


def test_get_full_chain_concatenates_all_expiries() -> None:
    e1 = date(2026, 6, 19)
    e2 = date(2026, 7, 17)
    chains = {
        e1: pd.DataFrame([_toy_row(e1, 100), _toy_row(e1, 110)], columns=CHAIN_COLUMNS),
        e2: pd.DataFrame([_toy_row(e2, 100)], columns=CHAIN_COLUMNS),
    }
    provider = _FakeProvider(chains)
    df = provider.get_full_chain("TST")
    assert len(df) == 3
    assert set(df["expiry"].unique()) == {e1, e2}


def test_get_full_chain_respects_max_expiries() -> None:
    expiries = [date(2026, m, 1) for m in (6, 7, 8, 9)]
    chains = {
        e: pd.DataFrame([_toy_row(e, 100.0)], columns=CHAIN_COLUMNS) for e in expiries
    }
    provider = _FakeProvider(chains)
    df = provider.get_full_chain("TST", max_expiries=2)
    assert set(df["expiry"].unique()) == set(expiries[:2])


def test_get_full_chain_returns_empty_when_no_expiries() -> None:
    provider = _FakeProvider({})
    df = provider.get_full_chain("TST")
    assert df.empty
    assert list(df.columns) == CHAIN_COLUMNS


def test_get_full_chain_drops_individual_failed_expiries() -> None:
    e1 = date(2026, 6, 19)
    e2 = date(2026, 7, 17)
    bad = pd.DataFrame([{"raise": True}])
    chains = {
        e1: pd.DataFrame([_toy_row(e1, 100.0)], columns=CHAIN_COLUMNS),
        e2: bad,
    }
    provider = _FakeProvider(chains)
    df = provider.get_full_chain("TST")
    # e2 raised; only e1 should remain.
    assert len(df) == 1
    assert df["expiry"].iloc[0] == e1


def test_provider_rate_limit_error_is_a_runtimeerror() -> None:
    assert issubclass(ProviderRateLimitError, RuntimeError)
