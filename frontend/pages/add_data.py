"""
SmartStock – Add New Data Page
For adding historical records with known sales figures.
These records build the rolling-average context used by the prediction pipeline.
"""

import streamlit as st
import requests
import os

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from components.forms import sale_record_form
from components.cards import success_alert, error_alert, warning_alert


def render():
    st.title("📥 Add New Store Record")
    st.caption(
        "Submit a **historical** day/store record with its actual sales figure. "
        "These records are stored in MongoDB and used by the prediction pipeline "
        "to compute 7-day and 30-day rolling averages for future forecasts."
    )

    st.info(
        "💡 **Why is Sales required here?**  \n"
        "The forecasting model needs rolling averages from past sales to predict future days. "
        "Every record you add here becomes part of that history. "
        "To predict a *future* date, use the **Predict Sales** page.",
        icon=None,
    )

    result = sale_record_form()

    if result:
        with st.spinner("Saving to database …"):
            try:
                resp = requests.post(f"{API_BASE}/new", json=result, timeout=10)
                if resp.status_code == 201:
                    body = resp.json()
                    success_alert(
                        f"Record saved for Store {result['store']} on {result['sale_date']} "
                        f"(Sales: £{result['sales']:,.0f}). Document ID: `{body['id']}`"
                    )
                    if body.get("warnings"):
                        for w in body["warnings"]:
                            warning_alert(w)
                elif resp.status_code == 422:
                    detail = resp.json().get("detail", resp.text)
                    error_alert(f"Validation error: {detail}")
                else:
                    error_alert(f"API error {resp.status_code}: {resp.text}")
            except requests.exceptions.ConnectionError:
                error_alert(
                    "Cannot connect to the backend. "
                    "Make sure it's running: `uvicorn app.main:app --reload`"
                )

    with st.expander("📄 Sample MongoDB Document Structure"):
        st.json({
            "store":                1,
            "sale_date":            "2023-07-15",
            "day_of_week":          6,
            "sales":                8450.0,
            "promo":                1,
            "school_holiday":       0,
            "state_holiday":        "0",
            "competition_distance": 1270.0,
            "store_type":           "a",
            "assortment":           "a",
            "created_at":           "2023-07-15",
        })