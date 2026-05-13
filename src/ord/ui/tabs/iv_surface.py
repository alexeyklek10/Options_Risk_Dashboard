"""IV Surface tab: 3D regular-grid surface plus per-expiry 2D smile."""

from __future__ import annotations

import streamlit as st

from ord.analytics.iv_surface import interpolated_surface, observed_surface
from ord.ui.charts import iv_smile_2d, iv_surface_3d
from ord.ui.context import AppContext


def render(ctx: AppContext) -> None:
    chain = ctx.filtered_chain
    if chain.empty:
        st.info("No options data available for the current filters.")
        return

    st.subheader("Implied-volatility surface")
    surf = interpolated_surface(chain)
    if surf is None:
        st.warning(
            "Not enough liquid (positive IV) points in the filtered chain to fit "
            "a 3D surface. Widen the strike or expiry window in the sidebar."
        )
    else:
        st.plotly_chart(iv_surface_3d(surf), use_container_width=True)

    st.subheader("Per-expiry smile")
    expiries = sorted(chain["expiry"].dropna().unique())
    if not expiries:
        return
    selected = st.selectbox(
        "Expiry",
        options=expiries,
        format_func=lambda d: d.date().isoformat() if hasattr(d, "date") else str(d),
    )
    expiry_chain = chain[chain["expiry"] == selected]
    scatter = observed_surface(expiry_chain)
    expiry_label = selected.date().isoformat() if hasattr(selected, "date") else str(selected)
    st.plotly_chart(iv_smile_2d(scatter, expiry_label), use_container_width=True)
