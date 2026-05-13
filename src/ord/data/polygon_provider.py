"""Polygon-backed implementation of ``DataProvider``.

Self-disables if ``POLYGON_API_KEY`` is unset. Uses
``/v3/snapshot/options/{ticker}`` for chain pulls.

Implemented in step 7 of the build sequence.
"""

from __future__ import annotations
