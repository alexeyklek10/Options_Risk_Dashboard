"""yfinance-backed implementation of :class:`DataProvider`.

Always available (no API key). Vendor-stale fields (zero open interest, NaN
implied vol) are left as ``None`` rather than imputed -- the engine's hand-rolled
IV solver in ``ord.pricing.iv_solver`` provides a clean recomputation path
elsewhere.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, ClassVar

import pandas as pd

from ord.data.base import (
    CHAIN_COLUMNS,
    DataProvider,
    ProviderName,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from ord.utils.time import days_to_expiry, utcnow


def _coerce_optional_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def _coerce_optional_int(value: Any) -> int | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return int(f)


def _midpoint(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None or bid <= 0.0 or ask <= 0.0:
        return None
    return (bid + ask) / 2.0


def _normalize_side(
    raw: pd.DataFrame,
    ticker: str,
    expiry: date,
    option_type: str,
    underlying_price: float,
    fetched_at: datetime,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dte = days_to_expiry(expiry)
    for _, r in raw.iterrows():
        bid = _coerce_optional_float(r.get("bid"))
        ask = _coerce_optional_float(r.get("ask"))
        rows.append(
            {
                "ticker": ticker,
                "expiry": expiry,
                "dte": dte,
                "strike": float(r["strike"]),
                "option_type": option_type,
                "bid": bid,
                "ask": ask,
                "mid": _midpoint(bid, ask),
                "last": _coerce_optional_float(r.get("lastPrice")),
                "volume": _coerce_optional_int(r.get("volume")),
                "open_interest": _coerce_optional_int(r.get("openInterest")),
                "implied_vol": _coerce_optional_float(r.get("impliedVolatility")),
                "underlying_price": underlying_price,
                "fetched_at": fetched_at,
                "source": "yfinance",
            }
        )
    return pd.DataFrame(rows, columns=CHAIN_COLUMNS)


class YFinanceProvider(DataProvider):
    """Default provider. No credentials required."""

    name: ClassVar[ProviderName] = "yfinance"

    def __init__(self) -> None:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - yfinance in requirements.txt
            raise ProviderUnavailableError("yfinance not installed") from exc
        self._yf = yf

    def get_underlying_price(self, ticker: str) -> float:
        t = self._yf.Ticker(ticker)
        # Prefer the live-quote ``fast_info`` path; fall back to last close.
        try:
            price = float(t.fast_info["last_price"])
            if price > 0:
                return price
        except Exception:  # noqa: BLE001 - fall through to history
            pass
        history = t.history(period="1d", interval="1d")
        if history.empty:
            raise ProviderRateLimitError(
                f"yfinance returned no data for {ticker} (rate limit or invalid ticker)"
            )
        return float(history["Close"].iloc[-1])

    def get_expirations(self, ticker: str) -> list[date]:
        t = self._yf.Ticker(ticker)
        raw = t.options
        if not raw:
            return []
        return [datetime.strptime(e, "%Y-%m-%d").date() for e in raw]

    def get_chain(self, ticker: str, expiry: date) -> pd.DataFrame:
        t = self._yf.Ticker(ticker)
        try:
            chain = t.option_chain(expiry.strftime("%Y-%m-%d"))
        except Exception as exc:  # noqa: BLE001 - normalize to our error
            raise ProviderRateLimitError(f"yfinance option_chain failed: {exc}") from exc
        underlying = self.get_underlying_price(ticker)
        fetched_at = utcnow()
        calls = _normalize_side(chain.calls, ticker, expiry, "call", underlying, fetched_at)
        puts = _normalize_side(chain.puts, ticker, expiry, "put", underlying, fetched_at)
        return pd.concat([calls, puts], ignore_index=True)
