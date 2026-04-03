"""
SmartStock – Card & Alert Components
"""

import streamlit as st


def prediction_card(predicted_sales: float, decision: str, reason: str, confidence: float):
    """Renders a styled prediction result card."""

    is_halt = decision == "Human Review"
    badge_color = "#e74c3c" if is_halt else "#27ae60"
    badge_icon  = "⚠️" if is_halt else "✅"

    st.markdown(
        f"""
        <div style="
            border-radius: 12px;
            border: 2px solid {badge_color};
            padding: 24px 28px;
            margin-top: 16px;
            background: linear-gradient(135deg, {'#fff5f5' if is_halt else '#f0fff4'}, white);
        ">
            <h2 style="margin:0; color:#2c3e50;">Prediction Result</h2>
            <hr style="border-color:{badge_color};"/>
            <div style="display:flex; gap:32px; flex-wrap:wrap;">
                <div>
                    <p style="margin:0; color:#7f8c8d; font-size:13px;">PREDICTED SALES</p>
                    <p style="margin:0; font-size:36px; font-weight:700; color:#2c3e50;">
                        £{predicted_sales:,.0f}
                    </p>
                </div>
                <div>
                    <p style="margin:0; color:#7f8c8d; font-size:13px;">DECISION</p>
                    <p style="margin:0; font-size:24px; font-weight:700; color:{badge_color};">
                        {badge_icon} {decision}
                    </p>
                </div>
                <div>
                    <p style="margin:0; color:#7f8c8d; font-size:13px;">MODEL CONFIDENCE</p>
                    <p style="margin:0; font-size:24px; font-weight:700; color:#2c3e50;">
                        {confidence:.0%}
                    </p>
                </div>
            </div>
            <div style="margin-top:16px; padding:12px; background:#f8f9fa; border-radius:8px;">
                <p style="margin:0; color:#555; font-size:14px;">
                    <strong>Reason:</strong> {reason}
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def success_alert(message: str):
    st.markdown(
        f"""<div style="background:#d4edda; border-left:5px solid #28a745;
            padding:12px 16px; border-radius:6px; color:#155724;">
            ✅ {message}</div>""",
        unsafe_allow_html=True,
    )


def error_alert(message: str):
    st.markdown(
        f"""<div style="background:#f8d7da; border-left:5px solid #dc3545;
            padding:12px 16px; border-radius:6px; color:#721c24;">
            ❌ {message}</div>""",
        unsafe_allow_html=True,
    )


def warning_alert(message: str):
    st.markdown(
        f"""<div style="background:#fff3cd; border-left:5px solid #ffc107;
            padding:12px 16px; border-radius:6px; color:#856404;">
            ⚠️ {message}</div>""",
        unsafe_allow_html=True,
    )


def info_card(title: str, value: str, icon: str = "ℹ️"):
    st.markdown(
        f"""<div style="background:#f0f4ff; border-radius:10px; padding:16px 20px;
            text-align:center; border:1px solid #c3d0f7;">
            <div style="font-size:28px;">{icon}</div>
            <div style="color:#7f8c8d; font-size:12px; margin-top:4px;">{title}</div>
            <div style="font-size:20px; font-weight:700; color:#2c3e50; margin-top:4px;">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )
