"""
SmartStock - Predict Sales Page
User fills in the future-day context; backend uses DB history for rolling stats.
"""

import streamlit as st
import requests
import os
from datetime import date, timedelta

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from components.cards import prediction_card, error_alert, info_card, warning_alert


def render():
    st.title("Predict Sales & Halt Decision")
    st.caption(
        "Enter the details for the **future day** you want to predict. "
        "The model uses your store's historical sales from the database to "
        "compute rolling averages — no actual sales needed for the target date."
    )

    st.markdown("### Store & Date")
    c1, c2, c3 = st.columns(3)
    store       = c1.number_input("Store ID", min_value=1, max_value=1115, value=1, step=1)
    # Default to tomorrow so it's clearly a future prediction
    sale_date   = c2.date_input("Predict For Date", value=date.today() + timedelta(days=1), min_value=date(2013, 1, 1))
    day_of_week = c3.selectbox(
        "Day of Week",
        [1, 2, 3, 4, 5, 6, 7],
        index=sale_date.weekday(),   # pre-fill from the chosen date (Mon=0 → 1)
        format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x - 1],
    )

    st.markdown("### Store Attributes")
    c4, c5, c6 = st.columns(3)
    store_type   = c4.selectbox("Store Type",  ["a", "b", "c", "d"])
    assortment   = c5.selectbox("Assortment",  ["a", "b", "c"])
    comp_dist    = c6.number_input(
        "Competition Distance (m)", min_value=0.0, max_value=200_000.0,
        value=1270.0, step=10.0,
    )

    st.markdown("### Day Flags")
    c7, c8, c9 = st.columns(3)
    promo          = c7.selectbox("Promo Active",   [0, 1], format_func=lambda x: "Yes" if x else "No")
    school_holiday = c8.selectbox("School Holiday", [0, 1], format_func=lambda x: "Yes" if x else "No")
    state_holiday  = c9.selectbox(
        "State Holiday", ["0", "a", "b", "c"],
        format_func=lambda x: {"0": "None", "a": "Public", "b": "Easter", "c": "Christmas"}[x],
    )

    st.divider()

    predict_btn = st.button("Predict Sales", use_container_width=True, type="primary")

    if predict_btn:
        payload = {
            "store":                int(store),
            "sale_date":            str(sale_date),
            "day_of_week":          int(day_of_week),
            "promo":                int(promo),
            "school_holiday":       int(school_holiday),
            "state_holiday":        state_holiday,
            "competition_distance": float(comp_dist),
            "store_type":           store_type,
            "assortment":           assortment,
        }

        with st.spinner("Fetching history & running prediction pipeline …"):
            try:
                resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=30)

                if resp.status_code == 200:
                    data = resp.json()

                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    with kpi1:
                        info_card("Store",       str(data["store"]),           "")
                    with kpi2:
                        info_card("Predict Date", data["sale_date"],            "")
                    with kpi3:
                        info_card("Predicted Sales", f"£{data['predicted_sales']:,.0f}", "")
                    with kpi4:
                        info_card("Decision",    data["halt_decision"],         "")

                    st.write("")
                    prediction_card(
                        predicted_sales = data["predicted_sales"],
                        decision        = data["halt_decision"],
                        reason          = data["reason"],
                        confidence      = data["confidence"],
                    )

                elif resp.status_code == 404:
                    error_alert(
                        f"No historical records found for Store {store} before {sale_date}. "
                        "Upload historical data first using the bulk upload script."
                    )
                elif resp.status_code == 422:
                    detail = resp.json().get("detail", resp.text)
                    error_alert(f"Validation error: {detail}")
                else:
                    error_alert(f"API error {resp.status_code}: {resp.text}")

            except requests.exceptions.ConnectionError:
                error_alert(
                    "Cannot connect to the backend. "
                    "Make sure it is running: `uvicorn app.main:app --reload`"
                )

    with st.expander("How this works"):
        st.markdown("""
        1. You fill in the store context for the **future day** you want to predict.
        2. The backend fetches the **last 30 historical records** for that store
           from MongoDB (all with `sale_date < your chosen date`).
        3. It computes **7-day and 30-day rolling averages and std** from those records.
        4. The **XGBRegressor** predicts sales using your inputs + rolling features.
        5. The **XGBClassifier** decides whether the prediction is reliable enough
           to automate, or whether a human should review it first.
        """)

    with st.expander("Sample API request body"):
        st.json({
            "store": 1,
            "sale_date": str(date.today() + timedelta(days=1)),
            "day_of_week": 2,
            "promo": 1,
            "school_holiday": 0,
            "state_holiday": "0",
            "competition_distance": 1270.0,
            "store_type": "a",
            "assortment": "a",
        })