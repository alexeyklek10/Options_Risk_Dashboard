"""Cross-source validator for multi-provider chain data.

For every (expiry, strike, option_type) row present in at least two
providers, compute provider-pair divergence metrics. Surfaces three views in
the Data Quality tab:

- aggregate disagreement rate by provider pair,
- per-strike biggest IV disagreement,
- histogram of |median_provider_iv - recomputed_iv| (calibration of our
  hand-rolled solver against the provider-supplied IVs).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ord.pricing.iv_solver import implied_vol


@dataclass(frozen=True)
class ValidationResult:
    """Bundle of per-row divergence metrics + provider-pair aggregates."""

    per_row: pd.DataFrame  # one row per (expiry, strike, type) seen in >=2 providers
    pair_aggregates: pd.DataFrame  # columns: pair, n_overlap, mean_iv_range, median_iv_range
    iv_calibration: pd.DataFrame  # columns: provider, abs_residual (per row)


def _pivot_to_one_row_per_contract(
    chains: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for provider, df in chains.items():
        if df.empty:
            continue
        sub = df[
            [
                "expiry",
                "strike",
                "option_type",
                "implied_vol",
                "mid",
                "open_interest",
                "underlying_price",
                "dte",
            ]
        ].copy()
        sub["provider"] = provider
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    return combined


def cross_validate(
    chains: dict[str, pd.DataFrame], r: float = 0.04, q: float = 0.0
) -> ValidationResult:
    """Compute divergence metrics across providers."""
    combined = _pivot_to_one_row_per_contract(chains)
    if combined.empty:
        return ValidationResult(
            per_row=pd.DataFrame(),
            pair_aggregates=pd.DataFrame(columns=["pair", "n_overlap", "mean_iv_range", "median_iv_range"]),
            iv_calibration=pd.DataFrame(columns=["provider", "abs_residual"]),
        )

    keys = ["expiry", "strike", "option_type"]
    grouped = combined.groupby(keys, dropna=False)
    per_row: list[dict[str, object]] = []
    calibration: list[dict[str, object]] = []
    pair_records: dict[tuple[str, str], list[float]] = {}

    for _, group in grouped:
        if len(group) < 2:
            continue
        ivs = group["implied_vol"].dropna()
        mids = group["mid"].dropna()
        ois = group["open_interest"].dropna()
        iv_range = float(ivs.max() - ivs.min()) if len(ivs) >= 2 else float("nan")
        mid_range = float(mids.max() - mids.min()) if len(mids) >= 2 else float("nan")
        oi_range = float(ois.max() - ois.min()) if len(ois) >= 2 else float("nan")
        median_iv = float(ivs.median()) if not ivs.empty else float("nan")

        # Recompute IV from the median mid via the hand-rolled solver and report
        # the calibration residual.
        recomputed_iv = None
        if not mids.empty:
            mid_for_solve = float(mids.median())
            S = float(group["underlying_price"].iloc[0])
            K = float(group["strike"].iloc[0])
            dte = int(group["dte"].iloc[0])
            T = max(dte, 1) / 365.0
            option_type = group["option_type"].iloc[0]
            recomputed_iv = implied_vol(mid_for_solve, S, K, T, r, option_type, q)
            if recomputed_iv is not None:
                for provider, sub in group.groupby("provider"):
                    iv = sub["implied_vol"].dropna()
                    if iv.empty:
                        continue
                    calibration.append(
                        {
                            "provider": provider,
                            "abs_residual": abs(float(iv.iloc[0]) - recomputed_iv),
                        }
                    )

        per_row.append(
            {
                "expiry": group["expiry"].iloc[0],
                "strike": float(group["strike"].iloc[0]),
                "option_type": group["option_type"].iloc[0],
                "median_iv": median_iv,
                "iv_range": iv_range,
                "mid_range": mid_range,
                "oi_range": oi_range,
                "recomputed_iv": recomputed_iv,
                "n_providers": len(group),
            }
        )

        providers = sorted(group["provider"].unique())
        for i, a in enumerate(providers):
            for b in providers[i + 1 :]:
                iv_a = group.loc[group["provider"] == a, "implied_vol"].dropna()
                iv_b = group.loc[group["provider"] == b, "implied_vol"].dropna()
                if iv_a.empty or iv_b.empty:
                    continue
                pair_records.setdefault((a, b), []).append(
                    abs(float(iv_a.iloc[0]) - float(iv_b.iloc[0]))
                )

    pair_rows = [
        {
            "pair": f"{a} vs {b}",
            "n_overlap": len(diffs),
            "mean_iv_range": float(np.mean(diffs)),
            "median_iv_range": float(np.median(diffs)),
        }
        for (a, b), diffs in pair_records.items()
    ]

    return ValidationResult(
        per_row=pd.DataFrame(per_row),
        pair_aggregates=pd.DataFrame(pair_rows),
        iv_calibration=pd.DataFrame(calibration),
    )
