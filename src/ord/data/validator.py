"""Cross-source validator for multi-provider chain data.

For each (expiry, strike, option_type) row present in at least two providers,
computes provider-pair divergence metrics: IV range, mid-price range, OI range,
and the gap between median provider IV and the IV recomputed from mid by the
hand-rolled solver. Powers the Data Quality tab.

Implemented in step 7 of the build sequence.
"""

from __future__ import annotations
