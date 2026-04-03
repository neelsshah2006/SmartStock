"""
SmartStock – Feature Engineering Utilities
Stateless helpers used by both preprocessing pipelines.
"""

import numpy as np
import pandas as pd
from typing import Sequence


def compute_rolling_stats(
    sales_history: Sequence[float],
    windows: tuple[int, ...] = (7, 30),
) -> dict[str, float]:
    """
    Given an ordered list of recent sales values (oldest → newest),
    return rolling mean and std for each requested window.

    Falls back gracefully when history is shorter than the window.
    """
    arr = np.array(sales_history, dtype=float)
    result: dict[str, float] = {}

    for w in windows:
        if len(arr) >= w:
            chunk = arr[-w:]
        else:
            chunk = arr  # use whatever we have

        result[f"average_sales_{w}"] = float(np.mean(chunk)) if len(chunk) > 0 else 0.0
        result[f"demand_std_{w}"]    = float(np.std(chunk,  ddof=1)) if len(chunk) > 1 else 0.0

    return result


def compute_halt_features(
    predicted_sales:   float,
    avg_sales_7:       float,
    avg_sales_30:      float,
    std_7:             float,
    std_30:            float,
    promo:             int,
    past_promo_sales:  float,
    past_normal_sales: float,
    days_since_promo:  int,
    forecast_confidence: float,
) -> dict[str, float]:
    """
    Reconstruct all halt-model features from known proxies.

    Since we have no actual future sales, we use avg_sales_7 as a proxy
    for 'Actual' wherever the training code required it.
    """
    # ── Forecast quality ────
    error_std_proxy = std_7  # best proxy without actuals
    fc = forecast_confidence if forecast_confidence > 0 else (1 / (1 + error_std_proxy))

    forecast_uncertainty   = std_7 / (predicted_sales + 1)
    forecast_risk          = (1 - fc) * _safe_div(std_7, avg_sales_7)

    # ── Promo features ──────
    current_uplift         = _safe_div(avg_sales_7 - avg_sales_7, avg_sales_7)  # 0 (no actual)
    expected_uplift        = _safe_div(past_promo_sales - past_normal_sales, past_normal_sales + 1)
    promotion_abnormality  = current_uplift - expected_uplift

    volatility_ratio       = _safe_div(std_7, avg_sales_7)
    promo_risk             = promo * volatility_ratio
    volatility_change      = _safe_div(std_7, std_30)
    promo_demand_risk      = promo * volatility_ratio

    # ── Trend / momentum ────
    trend_shift            = avg_sales_7 - avg_sales_30
    demand_momentum        = _safe_div(avg_sales_7 - avg_sales_30, avg_sales_30 + 1)

    # ── Shock / spike (proxy: compare avg_7 to avg_30 distribution) ─
    demand_shock_score     = _safe_div(avg_sales_7 - avg_sales_30, std_30 + 1)
    demand_spike           = int(avg_sales_7 > avg_sales_30 + 2 * std_30)
    demand_drop            = int(avg_sales_7 < avg_sales_30 - 2 * std_30)

    return {
        "Forecast_Confidence":    fc,
        "Forecast_Uncertainty":   forecast_uncertainty,
        "Promotion_Abnormality":  promotion_abnormality,
        "Promo_Risk":             promo_risk,
        "Past_Promo_Sales":       past_promo_sales,
        "Past_Normal_Sales":      past_normal_sales,
        "Days_Since_Last_Promo":  float(days_since_promo),
        "Volatility_Ratio":       volatility_ratio,
        "Volatility_Change":      volatility_change,
        "Trend_Shift":            trend_shift,
        "Demand_Shock_Score":     demand_shock_score,
        "Demand_Spike":           float(demand_spike),
        "Demand_Drop":            float(demand_drop),
        "Demand_Momentum":        demand_momentum,
        "Promo_Demand_Risk":      promo_demand_risk,
        "Forecast_Risk":          forecast_risk,
        "Predicted":              predicted_sales,
    }


def _safe_div(a: float, b: float) -> float:
    """Division that maps inf/nan → 0."""
    try:
        v = a / b if b != 0 else 0.0
        return 0.0 if (np.isinf(v) or np.isnan(v)) else v
    except Exception:
        return 0.0
