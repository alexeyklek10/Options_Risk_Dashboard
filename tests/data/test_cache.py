"""Parquet TTL cache: get / put / stale / clear semantics."""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from ord.data.base import CHAIN_COLUMNS, _empty_chain
from ord.data.cache import ChainCache


def _make_row(strike: float) -> dict[str, object]:
    return {
        "ticker": "TST",
        "expiry": date(2026, 6, 19),
        "dte": 37,
        "strike": strike,
        "option_type": "call",
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


@pytest.fixture()
def cache(tmp_path: Path) -> ChainCache:
    return ChainCache(tmp_path, ttl=timedelta(minutes=15))


def test_put_then_get_round_trips(cache: ChainCache) -> None:
    df = pd.DataFrame([_make_row(100.0), _make_row(110.0)], columns=CHAIN_COLUMNS)
    cache.put("yfinance", "TST", date(2026, 6, 19), df)
    cached = cache.get("yfinance", "TST", date(2026, 6, 19))
    assert cached is not None
    assert len(cached) == 2
    assert set(cached["strike"]) == {100.0, 110.0}


def test_get_returns_none_for_missing_key(cache: ChainCache) -> None:
    assert cache.get("yfinance", "TST", date(2026, 6, 19)) is None


def test_get_returns_none_when_stale(cache: ChainCache, tmp_path: Path) -> None:
    df = pd.DataFrame([_make_row(100.0)], columns=CHAIN_COLUMNS)
    cache.put("yfinance", "TST", date(2026, 6, 19), df)
    path = cache.path("yfinance", "TST", date(2026, 6, 19))
    # Backdate the file's mtime well past the TTL.
    old = time.time() - 3600
    os.utime(path, (old, old))
    assert cache.get("yfinance", "TST", date(2026, 6, 19)) is None


def test_put_ignores_empty_frame(cache: ChainCache) -> None:
    cache.put("yfinance", "TST", date(2026, 6, 19), _empty_chain())
    assert cache.get("yfinance", "TST", date(2026, 6, 19)) is None


def test_clear_removes_all_parquet_files(cache: ChainCache) -> None:
    df = pd.DataFrame([_make_row(100.0)], columns=CHAIN_COLUMNS)
    cache.put("yfinance", "TST", date(2026, 6, 19), df)
    cache.put("yfinance", "TST", date(2026, 7, 17), df)
    deleted = cache.clear()
    assert deleted == 2
    assert cache.get("yfinance", "TST", date(2026, 6, 19)) is None


def test_default_ttl_uses_market_aware_window(tmp_path: Path) -> None:
    # No fixed TTL passed; the cache should defer to default_cache_ttl.
    cache = ChainCache(tmp_path)
    df = pd.DataFrame([_make_row(100.0)], columns=CHAIN_COLUMNS)
    cache.put("yfinance", "TST", date(2026, 6, 19), df)
    # Fresh entry must be returned regardless of which window applies.
    assert cache.get("yfinance", "TST", date(2026, 6, 19)) is not None


def test_corrupted_parquet_treated_as_miss(tmp_path: Path) -> None:
    cache = ChainCache(tmp_path, ttl=timedelta(minutes=15))
    path = cache.path("yfinance", "TST", date(2026, 6, 19))
    path.write_text("not a parquet file")
    assert cache.get("yfinance", "TST", date(2026, 6, 19)) is None
