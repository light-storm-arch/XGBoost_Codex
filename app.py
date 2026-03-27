"""Streamlit entrypoint for ETF allocation app."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_SRC = Path(__file__).resolve().parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from etf_alloc_app.config.defaults import DEFAULT_SETTINGS
from etf_alloc_app.pipeline import run_pipeline


st.set_page_config(page_title="ETF Allocation App", layout="wide")
st.title("ETF Allocation App")
st.caption("Baseline scaffold with explicit configuration and pipeline orchestration.")

if st.button("Run baseline pipeline"):
    result = run_pipeline(DEFAULT_SETTINGS)
    st.success("Pipeline initialized")
    st.json(result)
