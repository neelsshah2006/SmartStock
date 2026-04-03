"""
SmartStock – Form Components
"""

import streamlit as st
from datetime import date


def sale_record_form() -> dict | None:
    """
    Renders the Add New Data form for historical records.
    The Sales field is required — this form stores past/known data
    so the prediction route can use it as rolling-stat context.
    Returns a dict ready to POST to /new, or None if not submitted.
    """
    with st.form("add_data_form", clear_on_submit=True):

        st.markdown("### 🏪 Store & Date")
        c1, c2, c3 = st.columns(3)
        store       = c1.number_input("Store ID", min_value=1, max_value=1115, value=1, step=1)
        sale_date   = c2.date_input("Sale Date", value=date.today(), min_value=date(2013, 1, 1))
        day_of_week = c3.selectbox(
            "Day of Week", [1, 2, 3, 4, 5, 6, 7],
            index=sale_date.weekday(),
            format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x - 1],
        )

        st.markdown("### 💰 Sales")
        sales = st.number_input(
            "Actual Sales",
            min_value=1.0,
            max_value=10_000_000.0,
            value=5000.0,
            step=10.0,
            help="The real sales figure for this day. Required so the model can build rolling averages for future predictions.",
        )

        st.markdown("### 📦 Store Attributes")
        c4, c5, c6 = st.columns(3)
        store_type = c4.selectbox("Store Type",  ["a","b","c","d"])
        assortment = c5.selectbox("Assortment",  ["a","b","c"])
        comp_dist  = c6.number_input(
            "Competition Distance (m)",
            min_value=0.0, max_value=200_000.0,
            value=1270.0, step=10.0,
        )

        st.markdown("### 🎯 Flags")
        c7, c8, c9 = st.columns(3)
        promo          = c7.selectbox("Promo Active",   [0, 1], format_func=lambda x: "Yes" if x else "No")
        school_holiday = c8.selectbox("School Holiday", [0, 1], format_func=lambda x: "Yes" if x else "No")
        state_holiday  = c9.selectbox(
            "State Holiday", ["0","a","b","c"],
            format_func=lambda x: {"0":"None","a":"Public","b":"Easter","c":"Christmas"}[x],
        )

        submitted = st.form_submit_button("💾  Save Record", use_container_width=True)

    if submitted:
        return {
            "store":                int(store),
            "sale_date":            str(sale_date),
            "day_of_week":          int(day_of_week),
            "sales":                float(sales),
            "promo":                int(promo),
            "school_holiday":       int(school_holiday),
            "state_holiday":        state_holiday,
            "competition_distance": float(comp_dist),
            "store_type":           store_type,
            "assortment":           assortment,
        }
    return None