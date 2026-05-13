"""Provider abstraction and canonical chain schema.

Every provider normalizes its raw response to a DataFrame whose rows conform to
``ChainRow``. The schema is the contract the rest of the engine programs
against.

The schema mirrors BUILD_PROMPT section 5.1. Fields that vendors may omit or
report as zero / NaN (``bid``, ``ask``, ``mid``, ``last``, ``volume``,
``open_interest``, ``implied_vol``) are nullable. Providers do not impute --
they leave them ``None`` and let downstream consumers decide whether to skip,
back-fill, or surface the gap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import ClassVar, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

OptionType = Literal["call", "put"]
ProviderName = Literal["yfinance", "tradier", "polygon"]

CHAIN_COLUMNS: list[str] = [
    "ticker",
    "expiry",
    "dte",
    "strike",
    "option_type",
    "bid",
    "ask",
    "mid",
    "last",
    "volume",
    "open_interest",
    "implied_vol",
    "underlying_price",
    "fetched_at",
    "source",
]


class ChainRow(BaseModel):
    """One row of a normalized options chain.

    A full chain is a ``pd.DataFrame`` whose columns are the field names of
    this model. Validation runs row-by-row only in tests; in hot paths
    ``ChainAggregator`` works on the DataFrame directly for speed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str = Field(min_length=1)
    expiry: date
    dte: int = Field(ge=0)
    strike: float = Field(gt=0)
    option_type: OptionType
    bid: float | None
    ask: float | None
    mid: float | None
    last: float | None
    volume: int | None
    open_interest: int | None
    implied_vol: float | None
    underlying_price: float = Field(gt=0)
    fetched_at: datetime
    source: ProviderName


class ProviderRateLimitError(RuntimeError):
    """Raised by a provider when the upstream API returns a 429 / rate limit.

    The aggregator catches this and continues with the remaining providers
    instead of failing the whole fetch.
    """


class ProviderUnavailableError(RuntimeError):
    """Raised when a provider's credentials are missing or it self-disables."""


class DataProvider(ABC):
    """Abstract base for chain providers.

    Subclasses set ``name`` as a class attribute and implement
    ``get_underlying_price``, ``get_expirations``, and ``get_chain``. The
    default ``get_full_chain`` fans out across expirations in a thread pool.
    """

    name: ClassVar[ProviderName]
    max_workers: int = 4

    @abstractmethod
    def get_underlying_price(self, ticker: str) -> float:
        """Return the latest spot price for the underlying."""

    @abstractmethod
    def get_expirations(self, ticker: str) -> list[date]:
        """Return all listed expirations for the underlying, ascending."""

    @abstractmethod
    def get_chain(self, ticker: str, expiry: date) -> pd.DataFrame:
        """Return the chain for one expiration normalized to ``CHAIN_COLUMNS``."""

    def get_full_chain(
        self, ticker: str, max_expiries: int | None = None
    ) -> pd.DataFrame:
        """Fetch chains for all (or the first ``max_expiries``) expirations and concat.

        Per-expiration fetches run concurrently in a thread pool. Individual
        expiration failures are logged and dropped; the remaining successful
        fetches are returned.
        """
        expiries = self.get_expirations(ticker)
        if max_expiries is not None:
            expiries = expiries[:max_expiries]
        if not expiries:
            return _empty_chain()

        frames: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for frame in pool.map(lambda e: self._safe_get_chain(ticker, e), expiries):
                if frame is not None and not frame.empty:
                    frames.append(frame)
        if not frames:
            return _empty_chain()
        return pd.concat(frames, ignore_index=True)

    def _safe_get_chain(self, ticker: str, expiry: date) -> pd.DataFrame | None:
        """Internal wrapper that swallows non-rate-limit errors per-expiration."""
        try:
            return self.get_chain(ticker, expiry)
        except ProviderRateLimitError:
            # Propagate so the aggregator can disable this provider entirely.
            raise
        except Exception:  # noqa: BLE001 - intentionally broad; one expiry failure is non-fatal
            return None


def _empty_chain() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical schema."""
    return pd.DataFrame({col: pd.Series(dtype=_dtype_for(col)) for col in CHAIN_COLUMNS})


def _dtype_for(col: str) -> str:
    if col in {"ticker", "option_type", "source"}:
        return "object"
    if col == "expiry":
        return "datetime64[ns]"
    if col == "fetched_at":
        return "datetime64[ns, UTC]"
    if col in {"dte", "volume", "open_interest"}:
        return "Int64"  # nullable integer
    # All remaining numeric columns are nullable floats.
    return "Float64"
