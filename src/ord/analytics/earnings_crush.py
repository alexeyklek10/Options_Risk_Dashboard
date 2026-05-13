"""Earnings IV crush estimator.

Decomposes the term structure of ATM IV across the two nearest expirations
that fall AFTER the announcement into a steady-state component and an
event component. Both expirations include the event, so their implied
variances satisfy::

    iv_short**2 * T_short = v_post * T_short + event_variance
    iv_long**2  * T_long  = v_post * T_long  + event_variance

where ``v_post`` is the annualized post-event steady-state variance. Solving
the two equations gives::

    v_post = (iv_long**2 * T_long - iv_short**2 * T_short) / (T_long - T_short)
    event_variance = iv_short**2 * T_short - v_post * T_short
    event_vol = sqrt(event_variance / T_event)

where ``T_event`` is calendar-time-to-announcement.

NOTE: The BUILD_PROMPT section 7.12 formula
``sqrt((iv_pre**2 * dte_pre - iv_post**2 * dte_post) / dte_event)`` evaluates
to ``sqrt(NEGATIVE)`` for any realistic post-event IV crush
(iv_pre > iv_post, dte_pre < dte_post), so it cannot be the intended
identity. We implement the two-equation form above instead, which is the
standard term-structure decomposition documented in event-vol literature
(e.g. SqueezeMetrics, Sinclair "Volatility Trading"). The deviation is
called out in the methodology notebook (step 8).

When historical price data is available we also report the realized 1-day
move on the most recent ``n_earnings`` announcements for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ord.analytics.skew import skew


@dataclass(frozen=True)
class EarningsCrushResult:
    """Earnings event-IV decomposition for one ticker."""

    earnings_date: date | None
    iv_pre: float | None  # IV of the FIRST expiry post-earnings (event-laden)
    iv_post: float | None  # IV of the NEXT expiry beyond that (further-out, also event-laden)
    dte_pre: int | None
    dte_post: int | None
    event_vol: float | None
    post_event_vol: float | None
    historical_moves: pd.Series


def _decompose(
    iv_pre: float,
    iv_post: float,
    t_pre: float,
    t_post: float,
    t_event: float,
) -> tuple[float | None, float | None]:
    if t_post <= t_pre or t_event <= 0.0:
        return None, None
    total_pre = iv_pre * iv_pre * t_pre
    total_post = iv_post * iv_post * t_post
    v_post = (total_post - total_pre) / (t_post - t_pre)
    if v_post < 0.0:
        # Calendar-arb input; cannot decompose cleanly.
        return None, None
    event_variance = total_pre - v_post * t_pre
    if event_variance <= 0.0:
        return None, float(np.sqrt(v_post))
    return float(np.sqrt(event_variance / t_event)), float(np.sqrt(v_post))


def _historical_moves(prices: pd.Series, earnings_dates: list[date], n: int) -> pd.Series:
    if prices.empty or not earnings_dates:
        return pd.Series(dtype="float64")
    sorted_ed = sorted(earnings_dates, reverse=True)
    moves: dict[date, float] = {}
    for ed in sorted_ed[:n]:
        ts = pd.Timestamp(ed)
        before_idx = prices.index.get_indexer([ts], method="ffill")
        if before_idx[0] < 0 or before_idx[0] >= len(prices) - 1:
            continue
        p0 = float(prices.iloc[before_idx[0]])
        p1 = float(prices.iloc[before_idx[0] + 1])
        if p0 <= 0:
            continue
        moves[ed] = (p1 - p0) / p0
    if not moves:
        return pd.Series(dtype="float64")
    return pd.Series(moves).sort_index(ascending=False)


def earnings_crush(
    chain: pd.DataFrame,
    earnings_date: date | None,
    as_of: date | None = None,
    underlying_history: pd.Series | None = None,
    historical_earnings_dates: list[date] | None = None,
    r: float = 0.04,
    q: float = 0.0,
    n_earnings: int = 4,
) -> EarningsCrushResult:
    """Estimate the implied event vol around the next earnings date."""
    history = _historical_moves(
        underlying_history if underlying_history is not None else pd.Series(dtype="float64"),
        historical_earnings_dates or [],
        n_earnings,
    )

    empty = EarningsCrushResult(
        earnings_date=earnings_date,
        iv_pre=None,
        iv_post=None,
        dte_pre=None,
        dte_post=None,
        event_vol=None,
        post_event_vol=None,
        historical_moves=history,
    )
    if earnings_date is None or chain.empty:
        return empty

    sk = skew(chain, r=r, q=q)
    if sk.per_expiry.empty:
        return empty

    per = sk.per_expiry.copy()
    dtes = chain.groupby("expiry")["dte"].first()
    per["dte"] = per["expiry"].map(dtes)

    # As-of date: from caller, otherwise infer from the chain's max fetched_at.
    if as_of is None and "fetched_at" in chain.columns:
        as_of_ts = chain["fetched_at"].max()
        as_of = as_of_ts.date() if hasattr(as_of_ts, "date") else date.today()
    if as_of is None:
        as_of = date.today()

    post = per[
        per["expiry"].apply(lambda e: (e.date() if hasattr(e, "date") else e) > earnings_date)
    ].dropna(subset=["atm_iv", "dte"])
    if len(post) < 2:
        return EarningsCrushResult(
            earnings_date=earnings_date,
            iv_pre=None,
            iv_post=None,
            dte_pre=None,
            dte_post=None,
            event_vol=None,
            post_event_vol=None,
            historical_moves=history,
        )

    pre_row = post.iloc[0]
    post_row = post.iloc[1]
    iv_pre = float(pre_row["atm_iv"])
    iv_post = float(post_row["atm_iv"])
    dte_pre = int(pre_row["dte"])
    dte_post = int(post_row["dte"])

    t_pre = max(dte_pre, 1) / 365.0
    t_post = max(dte_post, 1) / 365.0
    t_event = max((earnings_date - as_of).days, 0) / 365.0

    event_vol, post_event_vol = _decompose(iv_pre, iv_post, t_pre, t_post, t_event)
    return EarningsCrushResult(
        earnings_date=earnings_date,
        iv_pre=iv_pre,
        iv_post=iv_post,
        dte_pre=dte_pre,
        dte_post=dte_post,
        event_vol=event_vol,
        post_event_vol=post_event_vol,
        historical_moves=history,
    )
