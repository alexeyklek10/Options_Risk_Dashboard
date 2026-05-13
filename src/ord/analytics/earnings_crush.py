"""Earnings IV crush estimator.

Pulls next earnings date via yfinance ``Ticker.calendar``. Identifies the nearest
post-earnings expiry and the next expiry beyond it, then de-weights term structure
to back out the implied event vol::

    event_vol = sqrt((IV_pre**2 * DTE_pre - IV_post**2 * DTE_post) / DTE_event)

Also reports historical realized move across the past N earnings dates from the
price history.

Implemented in step 6 of the build sequence.
"""

from __future__ import annotations
