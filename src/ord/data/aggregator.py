"""Parallel chain aggregator across enabled providers.

Fans out provider fetches via ``ThreadPoolExecutor``, normalizes results to the
canonical ``ChainRow`` schema, and returns an ``AggregatedChain`` containing the
per-provider chains, a consensus chain (median IV per strike/expiry/type across
providers), and a discrepancy frame for the cross-source validator.

Implemented in steps 3 (yfinance-only) and 7 (multi-provider) of the build sequence.
"""

from __future__ import annotations
