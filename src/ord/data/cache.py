"""Parquet-backed TTL cache for provider chain pulls.

Keyed by ``(provider, ticker, expiry, fetched_date)``. TTL is configurable per
instance with sensible defaults (15 min intraday, 24 hr off-hours) provided by
``ord.utils.time.default_cache_ttl``.

The cache is purely opportunistic. Misses fall through silently; reads that hit
a stale entry return ``None`` so the caller re-fetches.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from ord.utils.time import default_cache_ttl, utcnow

_LOG = logging.getLogger(__name__)


def _cache_filename(provider: str, ticker: str, expiry: date) -> str:
    return f"{provider}__{ticker.upper()}__{expiry.isoformat()}.parquet"


class ChainCache:
    """Filesystem cache for provider chain DataFrames.

    Parameters
    ----------
    base_dir
        Directory to store parquet files in. Created if absent.
    ttl
        Optional fixed TTL. If ``None`` the cache uses
        :func:`ord.utils.time.default_cache_ttl` per request, which is
        15 minutes during NYSE session and 24 hours otherwise.
    """

    def __init__(self, base_dir: Path, ttl: timedelta | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._fixed_ttl = ttl

    def _ttl(self, now: datetime | None = None) -> timedelta:
        return self._fixed_ttl if self._fixed_ttl is not None else default_cache_ttl(now)

    def path(self, provider: str, ticker: str, expiry: date) -> Path:
        return self.base_dir / _cache_filename(provider, ticker, expiry)

    def get(
        self,
        provider: str,
        ticker: str,
        expiry: date,
        now: datetime | None = None,
    ) -> pd.DataFrame | None:
        """Return the cached frame, or ``None`` if it is missing or stale."""
        path = self.path(provider, ticker, expiry)
        if not path.exists():
            return None
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=utcnow().tzinfo)
        age = (now or utcnow()) - mtime
        if age > self._ttl(now):
            return None
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 - corrupted cache is best-effort
            _LOG.warning("Failed to read cache %s: %s; treating as miss", path, exc)
            return None

    def put(self, provider: str, ticker: str, expiry: date, df: pd.DataFrame) -> None:
        if df.empty:
            return
        path = self.path(provider, ticker, expiry)
        df.to_parquet(path, index=False)

    def clear(self) -> int:
        """Delete all parquet files in the cache directory. Returns deleted count."""
        deleted = 0
        for path in self.base_dir.glob("*.parquet"):
            path.unlink(missing_ok=True)
            deleted += 1
        return deleted
