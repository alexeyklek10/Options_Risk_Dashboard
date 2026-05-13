"""Data Quality tab: cross-source disagreement metrics + recomputed-IV calibration.

Implementation lands in step 7 of the build sequence (Tradier and Polygon
providers + the cross-source validator).
"""

from __future__ import annotations

import streamlit as st

from ord.ui.context import AppContext


def render(ctx: AppContext) -> None:
    if len(ctx.chains_by_provider) <= 1:
        st.info(
            "Data-quality cross-source view requires at least two providers. "
            "Set TRADIER_TOKEN or POLYGON_API_KEY in the environment to enable "
            "(wiring lands in step 7)."
        )
        return
    st.info("Cross-source validator view lands in step 7.")
