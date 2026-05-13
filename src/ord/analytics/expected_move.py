"""Expected-move cone derived from ATM straddle mid prices.

Uses the standard approximation ``expected_move ~= 0.85 * ATM_straddle_mid``
per expiry. The 0.85 factor is justified in the methodology notebook.

Implemented in step 4 of the build sequence.
"""

from __future__ import annotations
