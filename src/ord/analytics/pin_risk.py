"""Pin-risk likelihood score for the nearest weekly/monthly expiry.

Combines three signals: distance of strike from spot (within +/-2 percent),
open interest in the top decile of the chain, and total gamma exposure within
1 percent of the gamma flip level. Output is a 0-100 score per candidate strike.

Implemented in step 6 of the build sequence.
"""

from __future__ import annotations
