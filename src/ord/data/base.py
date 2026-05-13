"""Provider abstraction and canonical chain schema.

This module will define:
- ``ChainRow``: a pydantic model that every provider normalizes to.
- ``DataProvider``: an ABC with ``get_underlying_price``, ``get_expirations``,
  ``get_chain``, and a default ``get_full_chain`` that fans out across expirations.
- ``ProviderRateLimitError``: raised on 429s so the aggregator can skip a provider
  without failing the whole fetch.

Implemented in step 3 of the build sequence.
"""

from __future__ import annotations
