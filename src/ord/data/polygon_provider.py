"""Polygon.io-backed implementation of :class:`DataProvider`.

Self-disables when ``POLYGON_API_KEY`` is unset. Uses Polygon v3 snapshot
endpoints, which return a contract-level chain in a single call.

- ``/v3/snapshot/options/{ticker}``        - full chain with IV, OI, greeks.
- ``/v2/aggs/ticker/{ticker}/prev``        - previous close as a fallback.
- ``/v3/reference/options/contracts``      - listed contracts (not used in
  the snapshot path, but kept here for documentation).
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

_BASE = "https://api.polygon.io"


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


class PolygonProvider(DataProvider):
    """Polygon.io provider. Auto-disables when ``POLYGON_API_KEY`` is unset."""

    name: ClassVar[ProviderName] = "polygon"
    timeout: float = 12.0
    snapshot_limit: int = 250

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise ProviderUnavailableError("POLYGON_API_KEY env var is not set")
        self._session = requests.Session()
        self._session.params = {"apiKey": self.api_key}

    def _get(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        url = f"{_BASE}{path}"
        resp = self._session.get(url, params=params, timeout=self.timeout)
        if resp.status_code == 429:
            raise ProviderRateLimitError(f"Polygon rate-limited on {path}")
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result

    def get_underlying_price(self, ticker: str) -> float:
        data = self._get(f"/v2/aggs/ticker/{ticker}/prev", {})
        results = data.get("results")
        if not results:
            raise ProviderRateLimitError(f"Polygon returned no results for {ticker}")
        close = _coerce_float(results[0].get("c"))
        if close is None or close <= 0:
            raise ProviderRateLimitError(f"Polygon invalid close for {ticker}")
        return close

    def get_expirations(self, ticker: str) -> list[date]:
        # The snapshot endpoint returns all expirations as part of each contract;
        # we deduplicate from the chain rather than calling a separate endpoint.
        data = self._get(
            f"/v3/snapshot/options/{ticker}",
            {"limit": str(self.snapshot_limit)},
        )
        items = data.get("results", [])
        expiries: set[date] = set()
        for item in items:
            details = item.get("details", {})
            exp = details.get("expiration_date")
            if exp:
                expiries.add(datetime.strptime(exp, "%Y-%m-%d").date())
        return sorted(expiries)

    def get_chain(self, ticker: str, expiry: date) -> pd.DataFrame:
        data = self._get(
            f"/v3/snapshot/options/{ticker}",
            {
                "expiration_date": expiry.isoformat(),
                "limit": str(self.snapshot_limit),
            },
        )
        items = data.get("results", [])
        if not items:
            return pd.DataFrame(columns=CHAIN_COLUMNS)
        underlying = self.get_underlying_price(ticker)
        fetched_at = utcnow()
        dte = days_to_expiry(expiry)
        rows: list[dict[str, Any]] = []
        for item in items:
            details = item.get("details", {})
            day = item.get("day", {})
            quote = item.get("last_quote", {})
            iv = item.get("implied_volatility")
            opt_type_raw = (details.get("contract_type") or "").lower()
            if opt_type_raw not in {"call", "put"}:
                continue
            bid = _coerce_float(quote.get("bid"))
            ask = _coerce_float(quote.get("ask"))
            mid = None
            if bid is not None and ask is not None and bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
            rows.append(
                {
                    "ticker": ticker,
                    "expiry": expiry,
                    "dte": dte,
                    "strike": float(details["strike_price"]),
                    "option_type": opt_type_raw,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "last": _coerce_float(day.get("close")),
                    "volume": _coerce_int(day.get("volume")),
                    "open_interest": _coerce_int(item.get("open_interest")),
                    "implied_vol": _coerce_float(iv),
                    "underlying_price": underlying,
                    "fetched_at": fetched_at,
                    "source": "polygon",
                }
            )
        return pd.DataFrame(rows, columns=CHAIN_COLUMNS)
