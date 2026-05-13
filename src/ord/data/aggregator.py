"""Parallel chain aggregator.

Fans out chain fetches across one or more enabled providers and produces:

- ``chains``: one DataFrame per provider name.
- ``consensus``: a merged frame with median implied vol per (expiry, strike,
  option_type) across providers. With only one provider enabled (the default
  zero-secrets path), the consensus equals that provider's frame.
- ``discrepancies``: per-row provider-pair divergence metrics. Populated by
  :mod:`ord.data.validator` in step 7; an empty frame in step 3 when only
  yfinance is wired up.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import pandas as pd

from ord.data.base import (
    CHAIN_COLUMNS,
    DataProvider,
    ProviderRateLimitError,
    _empty_chain,
)
from ord.data.cache import ChainCache

_LOG = logging.getLogger(__name__)


@dataclass
class AggregatedChain:
    """Result bundle from :class:`ChainAggregator.fetch`."""

    chains: dict[str, pd.DataFrame] = field(default_factory=dict)
    consensus: pd.DataFrame = field(default_factory=_empty_chain)
    discrepancies: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    @property
    def is_empty(self) -> bool:
        return bool(self.consensus.empty)


class ChainAggregator:
    """Fan-out across enabled providers, with optional caching."""

    def __init__(
        self,
        providers: list[DataProvider],
        cache: ChainCache | None = None,
        max_workers: int = 4,
    ) -> None:
        if not providers:
            raise ValueError("ChainAggregator requires at least one provider")
        self.providers = providers
        self.cache = cache
        self.max_workers = max_workers

    def _fetch_one(
        self, provider: DataProvider, ticker: str, max_expiries: int | None
    ) -> pd.DataFrame:
        # Cache covers per-expiration parquet; the aggregator handles the
        # full-chain assembly. We let the provider fetch all expirations and
        # then split-and-cache below.
        chain = provider.get_full_chain(ticker, max_expiries=max_expiries)
        if self.cache is not None and not chain.empty:
            for expiry, group in chain.groupby("expiry"):
                exp_date = pd.Timestamp(expiry).date() if hasattr(expiry, "date") else expiry
                self.cache.put(provider.name, ticker, exp_date, group)
        return chain

    def fetch(self, ticker: str, max_expiries: int | None = None) -> AggregatedChain:
        """Fetch the chain from all enabled providers concurrently."""
        chains: dict[str, pd.DataFrame] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_provider = {
                pool.submit(self._fetch_one, p, ticker, max_expiries): p
                for p in self.providers
            }
            for fut in as_completed(future_to_provider):
                provider = future_to_provider[fut]
                try:
                    df = fut.result()
                    if not df.empty:
                        chains[provider.name] = df
                except ProviderRateLimitError as exc:
                    _LOG.warning(
                        "Provider %s rate-limited or unavailable: %s", provider.name, exc
                    )

        consensus = _build_consensus(chains)
        return AggregatedChain(chains=chains, consensus=consensus)


def _build_consensus(chains: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Median implied-vol per row across providers, with first-provider scalars elsewhere.

    With a single provider this is the provider's frame unchanged. With
    multiple providers it groups by ``(expiry, strike, option_type)``, takes
    the median implied vol across providers, and otherwise prefers the first
    provider's value per group. ``source`` is rewritten to ``"consensus"``.
    """
    if not chains:
        return _empty_chain()
    if len(chains) == 1:
        only = next(iter(chains.values()))
        return only.copy()

    combined = pd.concat(chains.values(), ignore_index=True)
    rows: list[dict[str, object]] = []
    for _key, group in combined.groupby(["expiry", "strike", "option_type"], dropna=False):
        row = group.iloc[0].to_dict()
        iv_vals = group["implied_vol"].dropna()
        row["implied_vol"] = float(iv_vals.median()) if not iv_vals.empty else None
        row["source"] = "consensus"
        rows.append(row)
    return pd.DataFrame(rows, columns=CHAIN_COLUMNS)
