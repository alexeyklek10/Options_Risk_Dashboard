"""Volatility skew: ATM IV, 25-delta risk reversal, 25-delta butterfly, near-ATM skew slope.

The 25-delta strikes are interpolated rather than rounded to the nearest listed
strike, so the metric is comparable across expiries with different listed grids.

Implemented in step 4 of the build sequence (skew_and_pcr tab).
"""

from __future__ import annotations
