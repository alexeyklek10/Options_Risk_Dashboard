# Architecture

One engine in `src/ord/` powers two front doors: the Streamlit dashboard (`streamlit_app.py`) and the methodology notebook (`notebooks/methodology.ipynb`). Neither front door duplicates analytics code.

```
[Data layer]    yfinance / Tradier / Polygon  ->  DataProvider ABC
                                              \
                                               -> ChainAggregator (parallel fetch, normalize)
                                                       \
                                                        -> CrossSourceValidator (divergence metrics)

[Pricing]       pricing/black_scholes.py
                pricing/greeks.py
                pricing/iv_solver.py
                (hand-rolled; py_vollib used only as a test oracle)

[Analytics]     iv_surface, greeks_dashboard, skew, max_pain, oi_heatmap,
                put_call_ratio, gex, iv_rv_spread, expected_move,
                implied_pdf, pin_risk, earnings_crush, strategy_builder

[Front doors]   Streamlit app  <->  Jupyter methodology notebook
```

## Module overview

| Module | Responsibility |
| --- | --- |
| `ord.data.base` | `ChainRow` pydantic schema; `DataProvider` ABC; `ProviderRateLimitError`. |
| `ord.data.yfinance_provider` | Default provider; no API key required. |
| `ord.data.tradier_provider` | Self-disables if `TRADIER_TOKEN` unset. |
| `ord.data.polygon_provider` | Self-disables if `POLYGON_API_KEY` unset. |
| `ord.data.aggregator` | Parallel fetch across enabled providers; consensus chain. |
| `ord.data.validator` | Cross-source divergence metrics for the Data Quality tab. |
| `ord.data.cache` | Parquet TTL cache under `data/cache/`. |
| `ord.pricing.black_scholes` | Scalar + vectorized BS pricer. |
| `ord.pricing.greeks` | Delta, gamma, vega, theta, rho, vanna, charm. |
| `ord.pricing.iv_solver` | Newton-Raphson with Brent fallback. |
| `ord.analytics.*` | One analytic per module, each with its own unit test. |
| `ord.ui.charts` | Shared Plotly chart factories. |
| `ord.ui.theme` | Color palette / layout defaults that mirror `.streamlit/config.toml`. |
| `ord.ui.tabs.*` | One Streamlit tab per file, each exposing `render(...)`. |
| `ord.utils.rates` | Risk-free rate fetcher: yfinance `^IRX`, optional FRED, hardcoded fallback. |
| `ord.utils.time` | Market session / DTE helpers. |

## Test layout

| Path | Coverage target |
| --- | --- |
| `tests/pricing/` | 100% |
| `tests/analytics/` | >= 85% |
| `tests/data/` | >= 70% |
| `src/ord/ui/` | Not measured (Streamlit code; manual + integration smoke). |

Provider integration tests are gated behind `@pytest.mark.skipif(not os.getenv("TRADIER_TOKEN"))` (and similarly for Polygon) so CI passes without secrets. The validator itself is fully tested using checked-in per-provider fixtures.
