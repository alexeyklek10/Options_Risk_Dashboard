"""Max-pain: strike that minimizes total dollar pain to option holders if the underlying pins.

Per-expiry argmin over strikes K of the standard pain function::

    pain(K) = sum over calls OI_i * max(K - K_i, 0)
            + sum over puts  OI_j * max(K_j - K, 0)

The methodology notebook is explicit that "stocks pin to max pain" is folklore.

Implemented in step 4 of the build sequence.
"""

from __future__ import annotations
