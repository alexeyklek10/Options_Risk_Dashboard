"""Tradier-backed implementation of :class:`DataProvider`.

Self-disables when ``TRADIER_TOKEN`` is unset (the aggregator silently skips
it). Defaults to the Tradier sandbox base URL; production base URL is used
when ``TRADIER_PRODUCTION=true``.

Tradier endpoints used:

- ``/v1/markets/quotes?symbols=...``       - latest underlying quote.
- ``/v1/markets/options/expirations``      - listed expirations.
- ``/v1/markets/options/chains?symbol=...`` - per-expiration chain.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import date, datetime
from typing import Any, ClassVar

import pandas as pd
import requests

from ord.data.base import (
    CHAIN_COLUMNS,
    DataProvider,
    ProviderName,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from ord.utils.time import days_to_expiry, utcnow

_LOG = logging.getLogger(__name__)

_SANDBOX_BASE = "https://sandbox.tradier.com"
_PRODUCTION_BASE = "https://api.tradier.com"


def _coerce_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def _coerce_int(value: Any) -> int | None:
    f = _coerce_float(value)
    if f is None:
        return None
    return int(f)


class TradierProvider(DataProvider):
    """Tradier provider. Auto-disables when ``TRADIER_TOKEN`` is unset."""

    name: ClassVar[ProviderName] = "tradier"
    timeout: float = 10.0

    def __init__(self, token: str | None = None, production: bool | None = None) -> None:
        self.token = token if token is not None else os.environ.get("TRADIER_TOKEN")
        if not self.token:
            raise ProviderUnavailableError("TRADIER_TOKEN env var is not set")
        env_prod = (
            production
            if production is not None
            else os.environ.get("TRADIER_PRODUCTION", "").lower() in {"1", "true", "yes"}
        )
        self.base_url = _PRODUCTION_BASE if env_prod else _SANDBOX_BASE
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
            }
        )

    def _get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = self._session.get(url, params=params, timeout=self.timeout)
        if resp.status_code == 429:
            raise ProviderRateLimitError(f"Tradier rate-limited on {path}")
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result

    def get_underlying_price(self, ticker: str) -> float:
        data = self._get("/v1/markets/quotes", {"symbols": ticker})
        quotes = data.get("quotes", {})
        quote = quotes.get("quote")
        if isinstance(quote, list):
            quote = quote[0] if quote else None
        if not isinstance(quote, dict):
            raise ProviderRateLimitError(f"Tradier returned no quote for {ticker}")
        last = _coerce_float(quote.get("last")) or _coerce_float(quote.get("close"))
        if last is None or last <= 0:
            raise ProviderRateLimitError(f"Tradier returned invalid last price for {ticker}")
        return last

    def get_expirations(self, ticker: str) -> list[date]:
        data = self._get(
            "/v1/markets/options/expirations",
            {"symbol": ticker, "includeAllRoots": "true", "strikes": "false"},
        )
        node = data.get("expirations", {})
        if not node:
            return []
        raw = node.get("date", [])
        if isinstance(raw, str):
            raw = [raw]
        return [datetime.strptime(d, "%Y-%m-%d").date() for d in raw]

    def get_chain(self, ticker: str, expiry: date) -> pd.DataFrame:
        data = self._get(
            "/v1/markets/options/chains",
            {
                "symbol": ticker,
                "expiration": expiry.isoformat(),
                "greeks": "false",
            },
        )
        options = data.get("options")
        if not options:
            return pd.DataFrame(columns=CHAIN_COLUMNS)
        raw = options.get("option", [])
        if isinstance(raw, dict):
            raw = [raw]
        underlying = self.get_underlying_price(ticker)
        fetched_at = utcnow()
        dte = days_to_expiry(expiry)
        rows: list[dict[str, Any]] = []
        for opt in raw:
            opt_type_raw = (opt.get("option_type") or "").lower()
            if opt_type_raw not in {"call", "put"}:
                continue
            bid = _coerce_float(opt.get("bid"))
            ask = _coerce_float(opt.get("ask"))
            mid = None
            if bid is not None and ask is not None and bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
            rows.append(
                {
                    "ticker": ticker,
                    "expiry": expiry,
                    "dte": dte,
                    "strike": float(opt["strike"]),
                    "option_type": opt_type_raw,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "last": _coerce_float(opt.get("last")),
                    "volume": _coerce_int(opt.get("volume")),
                    "open_interest": _coerce_int(opt.get("open_interest")),
                    "implied_vol": None,  # Tradier IV requires greeks=true (extra cost)
                    "underlying_price": underlying,
                    "fetched_at": fetched_at,
                    "source": "tradier",
                }
            )
        return pd.DataFrame(rows, columns=CHAIN_COLUMNS)
