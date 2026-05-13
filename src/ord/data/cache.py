"""Parquet-backed TTL cache for provider chain pulls.

Keyed by ``(provider, ticker, expiry, fetched_date)``. Default TTL is 15 minutes
intraday and 24 hours outside US market hours. Cache files live under
``data/cache/`` (gitignored).

Implemented in step 3 of the build sequence.
"""

from __future__ import annotations
