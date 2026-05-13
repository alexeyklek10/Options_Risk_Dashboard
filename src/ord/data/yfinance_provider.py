"""yfinance-backed implementation of ``DataProvider``.

Always available (no API key required). Treats vendor-stale fields
(zero open interest, NaN implied vol) as ``None`` rather than imputing.

Implemented in step 3 of the build sequence.
"""

from __future__ import annotations
