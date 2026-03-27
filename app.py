import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except Exception:
    from sklearn.ensemble import RandomForestRegressor

    HAS_XGBOOST = False


st.set_page_config(page_title="ETF Relative Return Forecaster", layout="wide")
st.title("ETF Pair Relative Return Forecasting Dashboard")
st.caption(
    "Educational prototype. Outputs are model estimates, not investment advice."
)


DEFAULT_FEATURES = [
    "term_spread",
    "credit_spread",
    "vix",
    "inflation_surprise",
    "momentum_1m",
    "momentum_3m",
]


@dataclass
class ModelResult:
    joined: pd.DataFrame
    summary: pd.DataFrame
    feature_importance: pd.DataFrame
    latest_snapshot: Dict[str, float]


def signal_bucket(value: float) -> str:
    abs_v = abs(value)
    if abs_v < 0.001:
        return "Neutral"
    if abs_v < 0.005:
        return "Weak"
    if abs_v < 0.01:
        return "Moderate"
    return "Strong"


def make_synthetic_data(n_months: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2010-01-31", periods=n_months, freq="M")

    df = pd.DataFrame(
        {
            "date": dates,
            "term_spread": rng.normal(0.01, 0.005, n_months).cumsum() / np.arange(1, n_months + 1),
            "credit_spread": rng.normal(0.015, 0.004, n_months),
            "vix": np.clip(rng.normal(20, 5, n_months), 10, 45),
            "inflation_surprise": rng.normal(0, 0.002, n_months),
            "momentum_1m": rng.normal(0, 0.03, n_months),
            "momentum_3m": rng.normal(0, 0.05, n_months),
        }
    )

    base_signal = (
        0.35 * df["term_spread"]
        - 0.25 * df["credit_spread"]
        - 0.001 * df["vix"]
        + 0.5 * df["momentum_1m"]
        + 0.25 * df["momentum_3m"]
        + 0.4 * df["inflation_surprise"]
    )

    df["spy_ret"] = 0.005 + base_signal + rng.normal(0, 0.02, n_months)
    df["tlt_ret"] = 0.003 - base_signal * 0.5 + rng.normal(0, 0.018, n_months)
    df["qqq_ret"] = 0.006 + base_signal * 1.25 + rng.normal(0, 0.028, n_months)
    df["ief_ret"] = 0.002 - base_signal * 0.35 + rng.normal(0, 0.014, n_months)
    return df


def build_features(
    df: pd.DataFrame,
    etf_a: str,
    etf_b: str,
    horizon: int,
    feature_cols: List[str],
    max_lag: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    work = df.copy().sort_values("date").reset_index(drop=True)
    rel_ret = work[f"{etf_a}_ret"] - work[f"{etf_b}_ret"]

    lagged = pd.DataFrame(index=work.index)
    for f in feature_cols:
        for lag in range(1, max_lag + 1):
            lagged[f"{f}_lag{lag}"] = work[f].shift(lag)

    target = rel_ret.shift(-horizon)
    full = pd.concat([work[["date"]], lagged, rel_ret.rename("rel_ret_t")], axis=1)
    mask = ~lagged.isna().any(axis=1) & target.notna()

    X = full.loc[mask, lagged.columns]
    y = target.loc[mask]
    dates = full.loc[mask, "date"]
    return X, y, dates


def fit_model(X_train, y_train, learning_rate, max_depth, n_estimators, subsample):
    if HAS_XGBOOST:
        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        ).fit(X_train, y_train)

    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        min_samples_leaf=3,
    ).fit(X_train, y_train)


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    learning_rate: float,
    max_depth: int,
    n_estimators: int,
    subsample: float,
    n_splits: int,
) -> ModelResult:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    preds = pd.Series(index=y.index, dtype=float)
    records = []
    fold_importance = []

    for fold, (tr, te) in enumerate(splitter.split(X), start=1):
        model = fit_model(
            X.iloc[tr],
            y.iloc[tr],
            learning_rate,
            max_depth,
            n_estimators,
            subsample,
        )
        y_hat = model.predict(X.iloc[te])
        preds.iloc[te] = y_hat

        fold_df = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        fold_df["fold"] = fold
        fold_importance.append(fold_df)

        records.append(
            {
                "fold": fold,
                "start": dates.iloc[te].min(),
                "end": dates.iloc[te].max(),
                "mae": mean_absolute_error(y.iloc[te], y_hat),
                "r2": r2_score(y.iloc[te], y_hat),
                "hit_rate": (np.sign(y_hat) == np.sign(y.iloc[te])).mean(),
            }
        )

    joined = pd.DataFrame(
        {
            "date": dates,
            "realized": y,
            "predicted": preds,
        }
    ).dropna()

    joined["tilt_impact"] = np.sign(joined["predicted"]) * joined["realized"]
    joined["cum_tilt_impact"] = joined["tilt_impact"].cumsum()

    summary = pd.DataFrame(records)

    feat_imp = (
        pd.concat(fold_importance)
        .groupby("feature", as_index=False)
        .agg(mean_importance=("importance", "mean"), std_importance=("importance", "std"))
        .sort_values("mean_importance", ascending=False)
    )

    latest = joined.iloc[-1].to_dict()
    latest["date"] = joined.iloc[-1]["date"]

    return ModelResult(joined, summary, feat_imp, latest)


with st.sidebar:
    st.header("1) Pair selector")
    st.caption("Choose the ETF spread to forecast as ETF A return minus ETF B return.")
    etf_options = ["spy", "tlt", "qqq", "ief"]
    c1, c2 = st.columns(2)
    etf_a = c1.selectbox("ETF A", etf_options, index=0, help="First ETF in A-B relative return.")
    etf_b = c2.selectbox("ETF B", etf_options, index=1, help="Second ETF subtracted from ETF A return.")

    st.header("2) Horizon selector")
    horizons = st.multiselect(
        "Forecast horizons (months)",
        options=[1, 3, 6, 12],
        default=[1, 3, 6],
        help="Target is forward relative return over selected horizon; future data is never used in features.",
    )

    st.header("3) Model hyperparameters")
    learning_rate = st.slider("Learning rate", 0.01, 0.4, 0.08, 0.01, help="Lower values are smoother but may underfit.")
    max_depth = st.slider("Tree depth", 2, 8, 4, 1, help="Higher depth increases complexity and overfit risk.")
    n_estimators = st.slider("Number of trees", 50, 600, 250, 25, help="More trees may improve fit but slow runtime.")
    subsample = st.slider("Row subsample", 0.5, 1.0, 0.9, 0.05, help="Stochastic sampling to reduce overfitting.")

    st.header("4) Backtest controls")
    n_splits = st.slider(
        "Walk-forward folds",
        3,
        10,
        6,
        1,
        help="TimeSeriesSplit only trains on earlier periods and tests on later periods (anti-look-ahead).",
    )
    max_lag = st.slider(
        "Max lag (months)",
        1,
        12,
        3,
        help="All features are lagged by at least one period to prevent look-ahead leakage.",
    )

    st.header("5) Feature / lag diagnostics")
    feature_cols = st.multiselect(
        "Macro/market features",
        options=DEFAULT_FEATURES,
        default=DEFAULT_FEATURES,
        help="Selected drivers are transformed into lagged predictors only.",
    )

uploaded = st.file_uploader(
    "Optional CSV input (must include date, feature columns, and *_ret columns)",
    type=["csv"],
    help="If omitted, synthetic sample data is used for demonstration.",
)

if uploaded is not None:
    data = pd.read_csv(uploaded)
    data["date"] = pd.to_datetime(data["date"])
else:
    data = make_synthetic_data()

missing = [f for f in feature_cols if f not in data.columns]
for ticker in [etf_a, etf_b]:
    if f"{ticker}_ret" not in data.columns:
        missing.append(f"{ticker}_ret")

if etf_a == etf_b:
    st.error("ETF A and ETF B must be different.")
    st.stop()
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()
if not horizons:
    st.error("Choose at least one horizon.")
    st.stop()

all_results = {}
for h in horizons:
    X, y, dates = build_features(data, etf_a, etf_b, h, feature_cols, max_lag)
    if len(X) < (n_splits + 1) * 5:
        st.warning(f"Not enough history for horizon {h}m with current lag/fold settings.")
        continue
    all_results[h] = walk_forward(
        X,
        y,
        dates,
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        subsample=subsample,
        n_splits=n_splits,
    )

if not all_results:
    st.stop()

st.subheader("Feature / lag diagnostics")
for h, result in all_results.items():
    with st.expander(f"Horizon {h}m diagnostics", expanded=False):
        corr = result.joined[["realized", "predicted"]].corr().iloc[0, 1]
        st.metric("Predicted/realized correlation", f"{corr:.3f}")
        lag_cols = [c for c in result.feature_importance["feature"] if "lag" in c]
        st.write(f"Lagged features used: {len(lag_cols)}")
        st.dataframe(result.feature_importance.head(10), use_container_width=True)

st.subheader("6) Prediction output")
selected_h = st.selectbox("Visualization horizon", sorted(all_results.keys()))
res = all_results[selected_h]

col_a, col_b = st.columns(2)
with col_a:
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=res.joined["date"], y=res.joined["realized"], name="Realized"))
    fig_ts.add_trace(go.Scatter(x=res.joined["date"], y=res.joined["predicted"], name="Predicted"))
    fig_ts.update_layout(title=f"Predicted vs realized relative return ({selected_h}m)", height=360)
    st.plotly_chart(fig_ts, use_container_width=True)

with col_b:
    fig_tilt = px.line(
        res.joined,
        x="date",
        y="cum_tilt_impact",
        title="Cumulative tilt impact (illustrative only)",
    )
    st.plotly_chart(fig_tilt, use_container_width=True)
    st.caption(
        "Caveat: this ignores trading costs, slippage, taxes, position sizing, and execution constraints."
    )

st.subheader("7) Evaluation dashboard")
c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mean_absolute_error(res.joined['realized'], res.joined['predicted']):.4f}")
c2.metric("R²", f"{r2_score(res.joined['realized'], res.joined['predicted']):.3f}")
c3.metric("Hit rate", f"{(np.sign(res.joined['predicted']) == np.sign(res.joined['realized'])).mean():.1%}")

st.markdown("**Walk-forward performance summary table**")
st.dataframe(res.summary, use_container_width=True)

st.markdown("**Feature importance (stability caveat)**")
st.caption("Importances can be unstable across regimes and collinear inputs; use as directional clues.")
fig_imp = px.bar(
    res.feature_importance.head(20).sort_values("mean_importance"),
    x="mean_importance",
    y="feature",
    orientation="h",
    error_x="std_importance",
)
fig_imp.update_layout(height=480)
st.plotly_chart(fig_imp, use_container_width=True)

st.subheader("Latest prediction snapshot")
rows = []
for h, result in sorted(all_results.items()):
    latest = result.latest_snapshot
    pred = latest["predicted"]
    rec = etf_a if pred >= 0 else etf_b
    rows.append(
        {
            "horizon_m": h,
            "recommended_etf": rec,
            "predicted_excess_return": pred,
            "signal_strength": signal_bucket(pred),
            "as_of_date": latest["date"],
        }
    )

snap = pd.DataFrame(rows)
st.dataframe(snap, use_container_width=True)

latest_features = []
for f in feature_cols:
    latest_features.append(
        {
            "feature": f,
            "latest_value": data[f].iloc[-1],
            "latest_lag1_value_used": data[f].shift(1).iloc[-1],
        }
    )
st.markdown("**Latest feature values used (lag-aware)**")
st.dataframe(pd.DataFrame(latest_features), use_container_width=True)

st.info(
    "Anti-look-ahead behavior: target uses forward return, while predictors are shifted by at least one period. "
    "Walk-forward folds preserve chronological order."
)
