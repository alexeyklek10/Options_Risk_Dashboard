"""Shared Plotly chart factories used across tabs.

Centralizes axis labelling (units always included: ``Strike ($)``,
``IV (annualized, %)``, ``GEX ($MM per 1%)``), the dark-theme defaults, and
the reusable surface / heatmap / bar / line builders.

Implemented in step 5 of the build sequence.
"""

from __future__ import annotations
