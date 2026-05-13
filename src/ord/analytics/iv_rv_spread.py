"""IV vs realized vol spread.

Compares ATM implied vol of the nearest-expiry-greater-than-21-DTE to the
21-day close-to-close annualized realized vol. Time series if historical IV
is cached; current snapshot only otherwise.

Implemented in step 6 of the build sequence.
"""

from __future__ import annotations
