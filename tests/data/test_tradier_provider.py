"""TradierProvider: unit tests with mocked HTTP + opt-in integration."""

from __future__ import annotations

import os
from datetime import date
from typing import Any

import pandas as pd
import pytest

from ord.data import tradier_provider as tp
from ord.data.base import (
    CHAIN_COLUMNS,
    ProviderRateLimitError,
    ProviderUnavailableError,
)


def test_init_raises_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRADIER_TOKEN", raising=False)
    with pytest.raises(ProviderUnavailableError):
        tp.TradierProvider()


def test_helpers_coerce_floats_and_ints() -> None:
    assert tp._coerce_float(float("nan")) is None
    assert tp._coerce_float("not-a-float") is None
    assert tp._coerce_float("1.5") == 1.5
    assert tp._coerce_int("nope") is None
    assert tp._coerce_int(3.9) == 3


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
    def fake_get(url: str, params: dict[str, str] | None = None, timeout: float = 10.0):  # noqa: ARG001
        for fragment, resp in routes.items():
            if fragment in url:
                return resp
        return _MockResp(404, {})

    return fake_get


def test_get_underlying_price(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADIER_TOKEN", "fake")
    provider = tp.TradierProvider()
    monkeypatch.setattr(
        provider._session,
        "get",
        _mock_get_factory({"/quotes": _MockResp(200, {"quotes": {"quote": {"last": 195.7}}})}),
    )
    assert provider.get_underlying_price("SPY") == 195.7


def test_get_chain_normalizes_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADIER_TOKEN", "fake")
    provider = tp.TradierProvider()
    payload_chain = {
        "options": {
            "option": [
                {
                    "option_type": "call",
                    "strike": 100.0,
                    "bid": 5.0,
                    "ask": 5.2,
                    "last": 5.1,
                    "volume": 100,
                    "open_interest": 1000,
                },
                {
                    "option_type": "put",
                    "strike": 100.0,
                    "bid": 4.0,
                    "ask": 4.2,
                    "last": 4.1,
                    "volume": 50,
                    "open_interest": 800,
                },
                {
                    "option_type": "weird",  # filtered out
                    "strike": 100.0,
                },
            ]
        }
    }
    routes = {
        "/quotes": _MockResp(200, {"quotes": {"quote": {"last": 100.0}}}),
        "/chains": _MockResp(200, payload_chain),
    }
    monkeypatch.setattr(provider._session, "get", _mock_get_factory(routes))
    df = provider.get_chain("SPY", date(2026, 6, 19))
    assert list(df.columns) == CHAIN_COLUMNS
    assert len(df) == 2
    assert set(df["option_type"]) == {"call", "put"}
    assert df["source"].iloc[0] == "tradier"


def test_429_raises_rate_limit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADIER_TOKEN", "fake")
    provider = tp.TradierProvider()
    monkeypatch.setattr(
        provider._session, "get", _mock_get_factory({"/quotes": _MockResp(429, {})})
    )
    with pytest.raises(ProviderRateLimitError):
        provider.get_underlying_price("SPY")


def test_get_expirations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADIER_TOKEN", "fake")
    provider = tp.TradierProvider()
    monkeypatch.setattr(
        provider._session,
        "get",
        _mock_get_factory(
            {"/expirations": _MockResp(200, {"expirations": {"date": ["2026-06-19", "2026-07-17"]}})}
        ),
    )
    assert provider.get_expirations("SPY") == [date(2026, 6, 19), date(2026, 7, 17)]


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("TRADIER_TOKEN"), reason="TRADIER_TOKEN not set")
def test_live_tradier_spy_chain() -> None:
    provider = tp.TradierProvider()
    expiries = provider.get_expirations("SPY")
    assert expiries
    df = provider.get_chain("SPY", expiries[0])
    assert not df.empty
