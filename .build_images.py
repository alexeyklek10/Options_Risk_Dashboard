"""Generate the static PNGs the README embeds.

Run from the repo root:

    .venv/Scripts/python.exe .build_images.py

Pulls the same SPY chain fixture the methodology notebook uses (so the images
stay reproducible) and writes four PNGs to ``images/``:

- ``dashboard_hero.png`` -- composite KPI tiles + skew + GEX bars, intended as
  the README's top-of-page screenshot stand-in.
- ``iv_surface_spy.png`` -- 3D-ish heatmap of the IV surface.
- ``gex_spy.png`` -- per-strike GEX bar chart with gamma-flip annotation.
- ``skew_spy.png`` -- ATM IV term structure with 25-delta RR overlay.

Re-run whenever the fixture is refreshed. The output PNGs are checked in.
This script is excluded from ruff / black to keep the linter config strict
without having to lint a one-off build helper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ord.analytics.gex import gamma_exposure
from ord.analytics.iv_surface import interpolated_surface
from ord.analytics.max_pain import max_pain_all_expiries
from ord.analytics.put_call_ratio import put_call_ratio
from ord.analytics.skew import skew

FIXTURE_DATE = "2026_05_13"
FIXTURE = Path("notebooks/fixtures") / f"spy_chain_{FIXTURE_DATE}.parquet"
IMG_DIR = Path("images")

DARK_BG = "#0E1117"
PANEL_BG = "#1A1F2B"
TEXT = "#E6E9EF"
PRIMARY = "#5BA3F5"
POSITIVE = "#3FB950"
NEGATIVE = "#F85149"
NEUTRAL = "#9CA3AF"


def _apply_dark(ax):
    ax.set_facecolor(DARK_BG)
    for spine in ax.spines.values():
        spine.set_color(NEUTRAL)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(True, color=NEUTRAL, alpha=0.15)


def main() -> None:
    IMG_DIR.mkdir(exist_ok=True)
    chain = pd.read_parquet(FIXTURE)
    spot = float(chain["underlying_price"].iloc[0])
    print(f"Loaded {len(chain)} rows, spot ${spot:.2f}")

    # --------------------- IV surface ---------------------
    surf = interpolated_surface(chain, side="call", n_strikes=40, n_dtes=20)
    if surf is not None:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
        im = ax.imshow(
            surf.iv * 100,
            aspect="auto",
            origin="lower",
            extent=[surf.strike_grid.min(), surf.strike_grid.max(), surf.dte_grid.min(), surf.dte_grid.max()],
            cmap="viridis",
        )
        ax.axvline(spot, color="white", linestyle="--", linewidth=1, label=f"Spot ${spot:.2f}")
        ax.set_xlabel("Strike ($)")
        ax.set_ylabel("Days to expiry")
        ax.set_title("SPY implied-volatility surface (calls)")
        ax.legend(loc="upper right", facecolor=PANEL_BG, edgecolor=NEUTRAL, labelcolor=TEXT)
        cb = fig.colorbar(im, ax=ax, label="IV (annualized, %)")
        cb.ax.yaxis.set_tick_params(color=TEXT)
        plt.setp(cb.ax.get_yticklabels(), color=TEXT)
        _apply_dark(ax)
        fig.tight_layout()
        fig.savefig(IMG_DIR / "iv_surface_spy.png", dpi=120, facecolor=DARK_BG)
        plt.close(fig)
        print("  iv_surface_spy.png")

    # --------------------- GEX ---------------------
    gex = gamma_exposure(chain)
    df = gex.per_strike.copy()
    df = df[(df["strike"] >= spot * 0.85) & (df["strike"] <= spot * 1.15)]
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in df["gex_total"]]
    ax.bar(df["strike"], df["gex_total"] / 1e6, color=colors, width=0.8 * np.diff(df["strike"]).mean() if len(df) > 1 else 1.0)
    ax.axhline(0, color=NEUTRAL, linewidth=0.8)
    ax.axvline(spot, color="white", linestyle="--", linewidth=1, label=f"Spot ${spot:.2f}")
    if gex.gamma_flip_strike is not None:
        ax.axvline(gex.gamma_flip_strike, color=PRIMARY, linestyle=":", linewidth=1.5,
                   label=f"Gamma flip ${gex.gamma_flip_strike:.2f}")
    ax.set_xlabel("Strike ($)")
    ax.set_ylabel("GEX ($MM per 1% move)")
    ax.set_title(f"SPY gamma exposure (total {gex.total_gex/1e6:+.1f} $MM per 1% move)")
    ax.legend(facecolor=PANEL_BG, edgecolor=NEUTRAL, labelcolor=TEXT)
    _apply_dark(ax)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "gex_spy.png", dpi=120, facecolor=DARK_BG)
    plt.close(fig)
    print("  gex_spy.png")

    # --------------------- Skew term structure ---------------------
    sk = skew(chain)
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.plot(sk.per_expiry["expiry"].astype(str), sk.per_expiry["atm_iv"] * 100,
            color=PRIMARY, marker="o", label="ATM IV (%)")
    ax.set_xlabel("Expiry")
    ax.set_ylabel("ATM IV (annualized, %)", color=PRIMARY)
    ax.tick_params(axis="y", labelcolor=PRIMARY)
    ax2 = ax.twinx()
    ax2.plot(sk.per_expiry["expiry"].astype(str), sk.per_expiry["rr_25d"] * 100,
             color=NEGATIVE, linestyle=":", marker="s", label="25-delta RR (%)")
    ax2.set_ylabel("25-delta RR (%)", color=NEGATIVE)
    ax2.tick_params(axis="y", labelcolor=NEGATIVE)
    ax.set_title("SPY skew term structure")
    _apply_dark(ax)
    ax2.set_facecolor(DARK_BG)
    for spine in ax2.spines.values():
        spine.set_color(NEUTRAL)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "skew_spy.png", dpi=120, facecolor=DARK_BG)
    plt.close(fig)
    print("  skew_spy.png")

    # --------------------- Hero composite ---------------------
    pcr = put_call_ratio(chain)
    pains = max_pain_all_expiries(chain)
    nearest_expiry = sorted(pains.keys())[0] if pains else None
    nearest_pain = pains[nearest_expiry].strike if nearest_expiry else float("nan")

    fig = plt.figure(figsize=(14, 7), facecolor=DARK_BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.6, 1.6])

    # KPI strip
    kpi_ax = fig.add_subplot(gs[0, :])
    kpi_ax.set_facecolor(DARK_BG)
    kpi_ax.axis("off")
    kpis = [
        ("Spot", f"${spot:,.2f}"),
        ("ATM IV (nearest)", f"{sk.per_expiry.iloc[0]['atm_iv']*100:.1f}%" if not sk.per_expiry.empty else "n/a"),
        ("PCR (OI)", f"{pcr.by_oi_aggregate:.2f}" if pcr.by_oi_aggregate is not None else "n/a"),
        ("Max pain (nearest)", f"${nearest_pain:,.2f}" if not np.isnan(nearest_pain) else "n/a"),
        ("Total GEX ($MM / 1%)", f"{gex.total_gex/1e6:+.1f}"),
        ("Gamma flip", f"${gex.gamma_flip_strike:,.2f}" if gex.gamma_flip_strike else "n/a"),
    ]
    for i, (label, value) in enumerate(kpis):
        x = 0.02 + i * (1 / len(kpis))
        kpi_ax.text(x, 0.85, label, fontsize=10, color=NEUTRAL, transform=kpi_ax.transAxes)
        kpi_ax.text(x, 0.30, value, fontsize=18, color=TEXT, transform=kpi_ax.transAxes, weight="bold")

    # Skew panel
    skew_ax = fig.add_subplot(gs[1, 0])
    skew_ax.plot(sk.per_expiry["expiry"].astype(str), sk.per_expiry["atm_iv"] * 100,
                 color=PRIMARY, marker="o")
    skew_ax.set_title("Skew term structure")
    skew_ax.set_ylabel("ATM IV (%)")
    skew_ax.tick_params(axis="x", labelrotation=30)
    _apply_dark(skew_ax)

    # GEX panel
    gex_ax = fig.add_subplot(gs[1, 1])
    colors_h = [POSITIVE if v >= 0 else NEGATIVE for v in df["gex_total"]]
    width_h = 0.8 * np.diff(df["strike"]).mean() if len(df) > 1 else 1.0
    gex_ax.bar(df["strike"], df["gex_total"] / 1e6, color=colors_h, width=width_h)
    gex_ax.axvline(spot, color="white", linestyle="--", linewidth=1)
    if gex.gamma_flip_strike is not None:
        gex_ax.axvline(gex.gamma_flip_strike, color=PRIMARY, linestyle=":", linewidth=1.5)
    gex_ax.set_title("Gamma exposure")
    gex_ax.set_xlabel("Strike")
    gex_ax.set_ylabel("$MM / 1%")
    _apply_dark(gex_ax)

    # PCR panel
    pcr_ax = fig.add_subplot(gs[1, 2])
    pe = pcr.per_expiry
    x = np.arange(len(pe))
    w = 0.4
    pcr_ax.bar(x - w/2, pe["pcr_volume"].fillna(0), width=w, color=PRIMARY, label="Volume")
    pcr_ax.bar(x + w/2, pe["pcr_oi"].fillna(0), width=w, color=NEUTRAL, label="OI")
    pcr_ax.set_xticks(x, [str(e) for e in pe["expiry"]], rotation=30)
    pcr_ax.set_title("Put/call ratio")
    pcr_ax.set_ylabel("PCR")
    pcr_ax.legend(facecolor=PANEL_BG, edgecolor=NEUTRAL, labelcolor=TEXT)
    _apply_dark(pcr_ax)

    fig.suptitle(f"Options Risk Dashboard -- SPY snapshot, {FIXTURE_DATE.replace('_', '-')}",
                 color=TEXT, fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(IMG_DIR / "dashboard_hero.png", dpi=120, facecolor=DARK_BG)
    plt.close(fig)
    print("  dashboard_hero.png")

    print(f"Done. Wrote {len(list(IMG_DIR.glob('*.png')))} PNGs to {IMG_DIR}/")


if __name__ == "__main__":
    main()
