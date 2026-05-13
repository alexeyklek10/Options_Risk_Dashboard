"""Market-time helpers: DTE, session-window detection, default cache TTL."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from ord.utils.time import (
    NY_TZ,
    days_to_expiry,
    default_cache_ttl,
    is_us_market_hours,
    utcnow,
)


def test_utcnow_is_timezone_aware() -> None:
    now = utcnow()
    assert now.tzinfo is not None


def test_days_to_expiry_basic() -> None:
    asof = date(2026, 1, 1)
    assert days_to_expiry(date(2026, 1, 8), asof=asof) == 7
    # Past expiry clips to zero (we never report negative DTE).
    assert days_to_expiry(date(2025, 12, 31), asof=asof) == 0
    # Same day is zero.
    assert days_to_expiry(asof, asof=asof) == 0


def test_is_us_market_hours_during_session() -> None:
    # Wednesday 2026-01-07 14:00 NY = inside the session.
    inside = datetime(2026, 1, 7, 14, 0, tzinfo=NY_TZ)
    assert is_us_market_hours(inside) is True


def test_is_us_market_hours_before_open() -> None:
    before = datetime(2026, 1, 7, 9, 0, tzinfo=NY_TZ)
    assert is_us_market_hours(before) is False


def test_is_us_market_hours_after_close() -> None:
    after = datetime(2026, 1, 7, 16, 30, tzinfo=NY_TZ)
    assert is_us_market_hours(after) is False


def test_is_us_market_hours_weekend() -> None:
    saturday = datetime(2026, 1, 10, 14, 0, tzinfo=NY_TZ)
    assert is_us_market_hours(saturday) is False


def test_is_us_market_hours_naive_input_is_treated_as_utc() -> None:
    # Wednesday 2026-01-07 19:00 UTC = 14:00 NY (post-DST).
    naive_utc = datetime(2026, 1, 7, 19, 0)
    assert is_us_market_hours(naive_utc) is True


def test_default_cache_ttl_intraday_vs_offhours() -> None:
    intraday = datetime(2026, 1, 7, 14, 0, tzinfo=NY_TZ)
    offhours = datetime(2026, 1, 7, 22, 0, tzinfo=NY_TZ)
    assert default_cache_ttl(intraday) == timedelta(minutes=15)
    assert default_cache_ttl(offhours) == timedelta(hours=24)


def test_is_us_market_hours_uses_now_when_called_without_args() -> None:
    # Smoke -- the boolean returned depends on wall clock, but the function
    # must not raise.
    result = is_us_market_hours()
    assert isinstance(result, bool)


def test_default_cache_ttl_without_args_returns_a_timedelta() -> None:
    ttl = default_cache_ttl()
    assert isinstance(ttl, timedelta)
    assert ttl > timedelta(0)


def test_utc_offsetnaive_now_explicit() -> None:
    # Cover the explicit UTC branch in is_us_market_hours.
    now_utc = datetime(2026, 1, 7, 19, 0, tzinfo=UTC)
    assert is_us_market_hours(now_utc) is True
