"""PolygonProvider: unit tests with mocked HTTP + opt-in integration."""

from __future__ import annotations

import os
from datetime import date
from typing import Any

import pandas as pd
import pytest

from ord.data import polygon_provider as pp
from ord.data.base import (
    CHAIN_COLUMNS,
    ProviderRateLimitError,
    ProviderUnavailableError,
)


def test_init_raises_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    with pytest.raises(ProviderUnavailableError):
        pp.PolygonProvider()


class _MockResp:
    def __init__(self, status: int, payload: dict[str, Any]):
        self.status_code = status
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400 and self.status_code != 429:
            raise RuntimeError(f"http {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


def _mock_get_factory(routes: dict[str, _MockResp]):
    def fake_get(url: str, params: dict[str, str] | None = None, timeout: float = 12.0):  # noqa: ARG001
        for fragment, resp in routes.items():
            if fragment in url:
                return resp
        return _MockResp(404, {})

    return fake_get


def test_get_underlying_price(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "fake")
    provider = pp.PolygonProvider()
    monkeypatch.setattr(
        provider._session,
        "get",
        _mock_get_factory({"/aggs/ticker": _MockResp(200, {"results": [{"c": 199.1}]})}),
    )
    assert provider.get_underlying_price("SPY") == 199.1


def test_get_expirations_deduplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "fake")
    provider = pp.PolygonProvider()
    payload = {
        "results": [
            {"details": {"expiration_date": "2026-06-19"}},
            {"details": {"expiration_date": "2026-06-19"}},
            {"details": {"expiration_date": "2026-07-17"}},
        ]
    }
    monkeypatch.setattr(
        provider._session,
        "get",
        _mock_get_factory({"/snapshot/options/SPY": _MockResp(200, payload)}),
    )
    assert provider.get_expirations("SPY") == [date(2026, 6, 19), date(2026, 7, 17)]


def test_get_chain_normalizes_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "fake")
    provider = pp.PolygonProvider()
    snapshot_payload = {
        "results": [
            {
                "details": {
                    "expiration_date": "2026-06-19",
                    "strike_price": 100.0,
                    "contract_type": "call",
                },
                "day": {"close": 5.1, "volume": 1000},
                "last_quote": {"bid": 5.0, "ask": 5.2},
                "implied_volatility": 0.21,
                "open_interest": 5000,
            },
            {
                "details": {
                    "expiration_date": "2026-06-19",
                    "strike_price": 100.0,
                    "contract_type": "put",
                },
                "day": {"close": 4.0, "volume": 500},
                "last_quote": {"bid": 3.9, "ask": 4.1},
                "implied_volatility": 0.23,
                "open_interest": 4000,
            },
        ]
    }
    routes = {
        "/snapshot/options/SPY": _MockResp(200, snapshot_payload),
        "/aggs/ticker": _MockResp(200, {"results": [{"c": 100.0}]}),
    }
    monkeypatch.setattr(provider._session, "get", _mock_get_factory(routes))
    df = provider.get_chain("SPY", date(2026, 6, 19))
    assert list(df.columns) == CHAIN_COLUMNS
    assert len(df) == 2
    assert df["implied_vol"].iloc[0] == pytest.approx(0.21)


def test_429_raises_rate_limit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "fake")
    provider = pp.PolygonProvider()
    monkeypatch.setattr(
        provider._session, "get", _mock_get_factory({"/aggs/ticker": _MockResp(429, {})})
    )
    with pytest.raises(ProviderRateLimitError):
        provider.get_underlying_price("SPY")


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("POLYGON_API_KEY"), reason="POLYGON_API_KEY not set")
def test_live_polygon_spy_chain() -> None:
    provider = pp.PolygonProvider()
    expiries = provider.get_expirations("SPY")
    assert expiries
    df = provider.get_chain("SPY", expiries[0])
    assert not df.empty
