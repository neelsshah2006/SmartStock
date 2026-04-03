"""
SmartStock – Streamlit Frontend Entry Point
Run: streamlit run app.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import streamlit as st
from pages import add_data, predict

st.set_page_config(
    page_title = "SmartStock",
    page_icon  = "📈",
    layout     = "wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/combo-chart--v2.png",
        width=64,
    )
    st.title("SmartStock")
    st.caption("ML-powered retail demand forecasting & ordering halt decisioning")
    st.divider()

    page = st.radio(
        "Navigate",
        ["➕  Add New Data", "🔮  Predict Sales"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(
        """
        **Models**
        - `XGBRegressor` → Sales Forecast
        - `XGBClassifier` → Halt Decision

        **Stack**
        - FastAPI · MongoDB
        - XGBoost · Streamlit
        """
    )

# ── Page Router ───────────────────────────────────────────────────────────────
if "Add New Data" in page:
    add_data.render()
else:
    predict.render()
