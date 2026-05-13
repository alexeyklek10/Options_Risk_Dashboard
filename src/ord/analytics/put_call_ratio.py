"""Put/call ratio: per-expiry and aggregate, by volume and by open interest.

Reports a 20-day rolling PCR if historical chains are available in the cache.

Implemented in step 4 of the build sequence.
"""

from __future__ import annotations
