"""Tradier-backed implementation of ``DataProvider``.

Self-disables if ``TRADIER_TOKEN`` is unset (the aggregator silently skips it).
Defaults to the Tradier sandbox base URL; production base URL is used when
``TRADIER_PRODUCTION=true``.

Implemented in step 7 of the build sequence.
"""

from __future__ import annotations
