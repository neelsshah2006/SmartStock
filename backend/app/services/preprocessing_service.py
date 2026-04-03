"""
SmartStock – Preprocessing Service
Two distinct pipelines: one for the forecasting model, one for the halt model.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from app.utils.encoders import encode_value
from app.utils.feature_engineering import compute_halt_features

logger = logging.getLogger(__name__)

FORECAST_FEATURES = [
    "Store", "DayOfWeek", "Promo", "SchoolHoliday",
    "StateHoliday", "Year", "Month", "Day", "WeekOfYear",
    "CompetitionDistance", "StoreType", "Assortment",
    "Average_Sales_7", "Average_Sales_30", "Demand_Std_7", "Demand_Std_30",
]

HALT_FEATURES = [
    "Forecast_Confidence", "Forecast_Uncertainty", "Promotion_Abnormality",
    "Promo_Risk", "Past_Promo_Sales", "Past_Normal_Sales", "Days_Since_Last_Promo",
    "Volatility_Ratio", "Volatility_Change", "Trend_Shift", "Demand_Shock_Score",
    "Demand_Spike", "Demand_Drop", "Demand_Momentum", "Promo_Demand_Risk",
    "Forecast_Risk", "Predicted",
]


def build_forecast_row(
    record: dict,
    avg_7: float,
    avg_30: float,
    std_7: float,
    std_30: float,
    encoders: dict[str, LabelEncoder] | None,
) -> pd.DataFrame:
    """
    Assemble a single-row DataFrame ready for XGBRegressor.predict().

    Parameters
    ----------
    record   : raw input dict (keys match SaleRecordInput field names)
    avg_7/30 : rolling averages from historical DB records
    std_7/30 : rolling standard deviations
    encoders : label encoders loaded from disk
    """
    sh_enc  = encode_value("StateHoliday", record["state_holiday"],  encoders)
    st_enc  = encode_value("StoreType",    record["store_type"],     encoders)
    asm_enc = encode_value("Assortment",   record["assortment"],     encoders)

    sale_date = pd.to_datetime(record["sale_date"])

    row = {
        "Store":               record["store"],
        "DayOfWeek":           record["day_of_week"],
        "Promo":               record["promo"],
        "SchoolHoliday":       record["school_holiday"],
        "StateHoliday":        sh_enc,
        "Year":                sale_date.year,
        "Month":               sale_date.month,
        "Day":                 sale_date.day,
        "WeekOfYear":          sale_date.isocalendar()[1],
        "CompetitionDistance": record["competition_distance"],
        "StoreType":           st_enc,
        "Assortment":          asm_enc,
        "Average_Sales_7":     avg_7,
        "Average_Sales_30":    avg_30,
        "Demand_Std_7":        std_7,
        "Demand_Std_30":       std_30,
    }
    return pd.DataFrame([row])[FORECAST_FEATURES]


def build_halt_row(
    predicted_sales:     float,
    avg_7:               float,
    avg_30:              float,
    std_7:               float,
    std_30:              float,
    promo:               int,
    past_promo_sales:    float,
    past_normal_sales:   float,
    days_since_promo:    int,
    forecast_confidence: float,
) -> pd.DataFrame:
    """
    Assemble a single-row DataFrame ready for XGBClassifier.predict().

    For future-day predictions we have no actual sales, so we substitute
    proxy values derived from historical statistics (see feature_engineering.py).
    """
    features = compute_halt_features(
        predicted_sales    = predicted_sales,
        avg_sales_7        = avg_7,
        avg_sales_30       = avg_30,
        std_7              = std_7,
        std_30             = std_30,
        promo              = promo,
        past_promo_sales   = past_promo_sales,
        past_normal_sales  = past_normal_sales,
        days_since_promo   = days_since_promo,
        forecast_confidence= forecast_confidence,
    )
    return pd.DataFrame([features])[HALT_FEATURES]
