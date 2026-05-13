"""Color palette + Plotly layout defaults that mirror ``.streamlit/config.toml``.

Centralizing these here means tabs build charts via :mod:`ord.ui.charts`
without needing to repeat color and font choices. Keep the palette in sync
with ``.streamlit/config.toml``.
"""

from __future__ import annotations

from typing import Any

PRIMARY = "#5BA3F5"
BACKGROUND = "#0E1117"
SECONDARY_BACKGROUND = "#1A1F2B"
TEXT = "#E6E9EF"
GRID = "#262B38"
POSITIVE = "#3FB950"
NEGATIVE = "#F85149"
NEUTRAL = "#9CA3AF"
CALL_COLOR = "#5BA3F5"
PUT_COLOR = "#F49AC1"

PLOTLY_LAYOUT: dict[str, Any] = {
    "paper_bgcolor": BACKGROUND,
    "plot_bgcolor": BACKGROUND,
    "font": {"color": TEXT, "family": "Inter, system-ui, sans-serif"},
    "xaxis": {"gridcolor": GRID, "zerolinecolor": GRID, "linecolor": GRID},
    "yaxis": {"gridcolor": GRID, "zerolinecolor": GRID, "linecolor": GRID},
    "margin": {"l": 50, "r": 30, "t": 50, "b": 50},
    "legend": {"bgcolor": SECONDARY_BACKGROUND, "bordercolor": GRID},
    "hoverlabel": {"bgcolor": SECONDARY_BACKGROUND, "font_color": TEXT},
}


def apply_layout(fig: Any) -> Any:
    """Apply the default dark layout to a Plotly figure in-place and return it."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig
