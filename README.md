# ETF Relative Return Forecasting Dashboard

A Streamlit app for forecasting **relative return** between two ETFs (ETF A minus ETF B) across multiple horizons with walk-forward validation and model diagnostics.

> **Risk warning:** This project is for research and education. It is not investment advice, and model outputs can fail in live markets.

## Environment requirements

- Python 3.10+
- Recommended packages:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `plotly`
  - `matplotlib` (optional)
  - `xgboost` (optional but preferred)

Install quickly:

```bash
python -m venv .venv
source .venv/bin/activate
pip install streamlit pandas numpy scikit-learn plotly matplotlib xgboost
```

If `xgboost` is unavailable, the app falls back to `RandomForestRegressor`.

## Setup and run

1. Clone/open the repo.
2. Create and activate a virtual environment.
3. Install dependencies above.
4. Run:

```bash
streamlit run app.py
```

5. Open the local URL printed by Streamlit (typically `http://localhost:8501`).

## FRED key setup

If you plan to replace synthetic/sample data with FRED-sourced macro series:

1. Request a FRED API key from St. Louis Fed.
2. Export it in your shell:

```bash
export FRED_API_KEY="your_key_here"
```

3. In your custom data pipeline, load this environment variable and use it in API requests.

> The current `app.py` does not directly call FRED; it expects a prepared CSV or uses built-in synthetic data.

## Data pipeline and lag assumptions

Expected input CSV columns:

- `date`
- feature columns (default examples in app):
  - `term_spread`, `credit_spread`, `vix`, `inflation_surprise`, `momentum_1m`, `momentum_3m`
- ETF return columns named `<ticker>_ret` (e.g., `spy_ret`, `tlt_ret`, `qqq_ret`, `ief_ret`)

Pipeline behavior:

1. Compute contemporaneous relative return: `rel_ret_t = etf_a_ret - etf_b_ret`.
2. Build lagged features only (`lag1 ... lagN`).
3. Set target as forward return over selected horizon.
4. Drop rows with insufficient lag history or unavailable forward target.

### Anti-look-ahead policy

- **No current-period or future features** are used for prediction.
- Predictors are shifted by at least one period.
- Walk-forward validation only trains on historical windows and tests on later windows.

## Target definition

For horizon `h` months:

- `target_t = (ETF_A - ETF_B) forward relative return at t+h`
- Positive prediction ⇒ favor ETF A versus ETF B.
- Negative prediction ⇒ favor ETF B versus ETF A.

## Walk-forward methodology

- Uses `TimeSeriesSplit` with user-selected fold count.
- Each fold:
  - train on earlier dates,
  - test on later dates,
  - save out-of-sample predictions,
  - compute fold metrics (`MAE`, `R²`, hit rate).
- Aggregates fold-level metrics into the walk-forward summary table.

## Dashboard outputs

- Pair selector
- Horizon selector
- Hyperparameter controls
- Backtest controls
- Feature/lag diagnostics
- Prediction output chart (predicted vs realized)
- Cumulative tilt impact chart (illustrative only)
- Walk-forward summary table
- Feature importance chart with fold-level variability
- Latest prediction snapshot:
  - recommended ETF per horizon,
  - predicted excess return,
  - signal bucket (Neutral/Weak/Moderate/Strong),
  - latest feature values used (lag-aware)

## Interpretation guidance and limitations

- Treat feature importances as **unstable regime-dependent diagnostics**, not causal truth.
- High backtest fit can still fail out-of-sample due to structural breaks.
- Cumulative tilt impact is simplified and excludes:
  - transaction costs,
  - slippage,
  - taxes,
  - liquidity/execution constraints,
  - portfolio construction details.
- Signals should be used as one input in a broader risk framework.

## Risk warnings

- Model risk: misspecification, overfitting, and data revision bias.
- Data risk: stale, revised, or survivorship-biased inputs.
- Market risk: unexpected macro shocks and regime changes.
- Operational risk: implementation and execution differences from backtests.

Use this tooling responsibly and validate any strategy with independent controls before live deployment.
