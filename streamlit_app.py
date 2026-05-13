"""Entry point for the Options Risk Dashboard Streamlit app.

Streamlit Community Cloud runs this module directly. The full UI is assembled
from ``ord.ui.tabs.*`` and ``ord.ui.charts``; this file is intentionally thin.

The dashboard is implemented incrementally across build steps 5 (first six tabs:
Overview, IV Surface, Greeks, Skew and PCR, Max Pain and OI, GEX) and 6
(remaining tabs: IV vs RV, Implied PDF, Pin Risk, Strategy Builder) and 7
(Data Quality).
"""

from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="Options Risk Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Options Risk Dashboard")
    st.write(
        "Scaffold in place. The interactive dashboard is implemented incrementally "
        "across build steps 5 through 7. See the README for current status."
    )


if __name__ == "__main__":
    main()
