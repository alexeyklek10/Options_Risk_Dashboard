"""Implied-volatility surface across strikes and days-to-expiry.

Produces both a raw scatter (irregular grid of observed IVs) and a regularized
grid interpolated with cubic ``scipy.interpolate.griddata`` for 3D rendering.

Implemented in step 4 of the build sequence.
"""

from __future__ import annotations
