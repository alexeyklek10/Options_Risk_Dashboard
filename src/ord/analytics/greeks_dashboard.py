"""Per-strike, per-expiry Greeks recomputed from the hand-rolled pricer.

Returns a DataFrame attached to the chain with all seven Greeks: delta, gamma,
vega, theta, rho, vanna, charm.

Implemented in step 4 of the build sequence.
"""

from __future__ import annotations
