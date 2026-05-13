"""Risk-free rate fetcher with graceful fallbacks.

Resolution order:

1. Primary: yfinance ``^IRX`` (13-week T-bill yield, quoted as percent; divided by 100).
2. Optional: FRED ``DGS3MO`` if ``FRED_API_KEY`` is set.
3. Fallback: hardcoded 0.04 with a one-time logged warning.

This keeps the app's default path free of any required API keys.

Implemented in step 3 of the build sequence.
"""

from __future__ import annotations
