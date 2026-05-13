"""Aggregator fan-out and consensus construction."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from ord.data.aggregator import AggregatedChain, ChainAggregator
from ord.data.base import (
    CHAIN_COLUMNS,
    DataProvider,
    ProviderName,
    ProviderRateLimitError,
    _empty_chain,
)
from ord.data.cache import ChainCache


def _row(
    strike: float,
    expiry: date,
    iv: float | None,
    source: str = "yfinance",
    option_type: str = "call",
) -> dict[str, object]:
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
        "implied_vol": iv,
        "underlying_price": 100.0,
        "fetched_at": datetime(2026, 5, 13, 14, 30, tzinfo=timezone.utc),
        "source": source,
    }


class _StaticProvider(DataProvider):
    name: ClassVar[ProviderName]  # set per subclass

    def __init__(self, full_chain: pd.DataFrame, expiries: list[date]) -> None:
        self._full = full_chain
        self._expiries = expiries

    def get_underlying_price(self, ticker: str) -> float:  # noqa: ARG002
        return 100.0

    def get_expirations(self, ticker: str) -> list[date]:  # noqa: ARG002
        return self._expiries

    def get_chain(self, ticker: str, expiry: date) -> pd.DataFrame:  # noqa: ARG002
        return self._full[self._full["expiry"] == expiry].copy()

    def get_full_chain(  # noqa: D401 - override to bypass thread pool in tests
        self, ticker: str, max_expiries: int | None = None  # noqa: ARG002
    ) -> pd.DataFrame:
        return self._full.copy()


class _YfStub(_StaticProvider):
    name = "yfinance"


class _TradierStub(_StaticProvider):
    name = "tradier"


class _RateLimited(DataProvider):
    name: ClassVar[ProviderName] = "tradier"

    def get_underlying_price(self, ticker: str) -> float:  # noqa: ARG002
        return 100.0

    def get_expirations(self, ticker: str) -> list[date]:  # noqa: ARG002
        return [date(2026, 6, 19)]

    def get_chain(self, ticker: str, expiry: date) -> pd.DataFrame:  # noqa: ARG002
        raise ProviderRateLimitError("synthetic 429")

    def get_full_chain(
        self, ticker: str, max_expiries: int | None = None  # noqa: ARG002
    ) -> pd.DataFrame:
        raise ProviderRateLimitError("synthetic 429")


def test_requires_at_least_one_provider() -> None:
    with pytest.raises(ValueError):
        ChainAggregator(providers=[])


def test_single_provider_consensus_equals_provider_frame() -> None:
    e1 = date(2026, 6, 19)
    df = pd.DataFrame([_row(100.0, e1, 0.2), _row(110.0, e1, 0.25)], columns=CHAIN_COLUMNS)
    yf = _YfStub(df, [e1])
    agg = ChainAggregator(providers=[yf]).fetch("TST")
    assert not agg.is_empty
    assert len(agg.chains) == 1
    assert agg.consensus.equals(df)


def test_multi_provider_consensus_takes_median_iv() -> None:
    e1 = date(2026, 6, 19)
    yf_df = pd.DataFrame([_row(100.0, e1, 0.20, "yfinance")], columns=CHAIN_COLUMNS)
    tr_df = pd.DataFrame(
        [_row(100.0, e1, 0.22, "tradier")], columns=CHAIN_COLUMNS
    )
    yf = _YfStub(yf_df, [e1])
    tr = _TradierStub(tr_df, [e1])
    agg = ChainAggregator(providers=[yf, tr]).fetch("TST")
    assert agg.consensus.shape[0] == 1
    assert agg.consensus["implied_vol"].iloc[0] == pytest.approx(0.21)
    assert agg.consensus["source"].iloc[0] == "consensus"


def test_rate_limited_provider_is_skipped(caplog: pytest.LogCaptureFixture) -> None:
    e1 = date(2026, 6, 19)
    yf_df = pd.DataFrame([_row(100.0, e1, 0.20)], columns=CHAIN_COLUMNS)
    yf = _YfStub(yf_df, [e1])
    tr = _RateLimited()
    with caplog.at_level("WARNING", logger="ord.data.aggregator"):
        agg = ChainAggregator(providers=[yf, tr]).fetch("TST")
    assert "yfinance" in agg.chains
    assert "tradier" not in agg.chains
    assert any("rate-limited" in r.message for r in caplog.records)


def test_empty_provider_response_yields_empty_aggregate() -> None:
    yf = _YfStub(_empty_chain(), [])
    agg = ChainAggregator(providers=[yf]).fetch("TST")
    assert agg.is_empty


def test_aggregator_writes_per_expiry_cache(tmp_path: Path) -> None:
    e1 = date(2026, 6, 19)
    e2 = date(2026, 7, 17)
    df = pd.DataFrame(
        [_row(100.0, e1, 0.2), _row(110.0, e2, 0.25)], columns=CHAIN_COLUMNS
    )
    yf = _YfStub(df, [e1, e2])
    cache = ChainCache(tmp_path)
    ChainAggregator(providers=[yf], cache=cache).fetch("TST")
    assert cache.get("yfinance", "TST", e1) is not None
    assert cache.get("yfinance", "TST", e2) is not None


def test_aggregated_chain_dataclass_defaults() -> None:
    bundle = AggregatedChain()
    assert bundle.chains == {}
    assert bundle.is_empty
    assert bundle.discrepancies.empty
