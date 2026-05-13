"""Market-time helpers: DTE math, US equity session window check, UTC helpers.

Kept intentionally minimal -- the dashboard cares only about coarse "is the
market open right now" decisions (for cache-TTL selection) and DTE computed
calendar-day style.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta, tzinfo
from zoneinfo import ZoneInfo

NY_TZ: tzinfo = ZoneInfo("America/New_York")
SESSION_OPEN: time = time(9, 30)
SESSION_CLOSE: time = time(16, 0)


def utcnow() -> datetime:
    """Timezone-aware UTC now (replaces deprecated naive ``datetime.utcnow``)."""
    return datetime.now(tz=UTC)


def days_to_expiry(expiry: date, asof: date | None = None) -> int:
    """Calendar DTE (BUILD_PROMPT section 5.1 convention). Same-day returns 0, never negative."""
    asof = asof if asof is not None else datetime.now(tz=NY_TZ).date()
    delta = (expiry - asof).days
    return max(delta, 0)


def is_us_market_hours(now: datetime | None = None) -> bool:
    """True if ``now`` (UTC or naive-assumed-UTC) falls inside the regular NYSE session.

    Does NOT check the NYSE holiday calendar -- this is good enough for cache-TTL
    selection (the worst case is a slightly longer TTL on a market holiday).
    """
    if now is None:
        now_ny = datetime.now(tz=NY_TZ)
    elif now.tzinfo is None:
        now_ny = now.replace(tzinfo=UTC).astimezone(NY_TZ)
    else:
        now_ny = now.astimezone(NY_TZ)
    if now_ny.weekday() >= 5:  # Sat/Sun
        return False
    return SESSION_OPEN <= now_ny.time() <= SESSION_CLOSE


def default_cache_ttl(now: datetime | None = None) -> timedelta:
    """15 minutes during the NYSE session, 24 hours otherwise."""
    return timedelta(minutes=15) if is_us_market_hours(now) else timedelta(hours=24)
