"""
SmartStock – Prediction Route
POST /predict  →  full two-stage pipeline: forecast → halt decision

The caller supplies the store context + future date they want to predict.
MongoDB is used ONLY to fetch historical sales for rolling-stat computation.
No 'sales' field is required — that is what we are predicting.
"""

import logging
from fastapi import APIRouter, HTTPException, status

from app.database.mongodb import get_db
from app.schemas.schemas import PredictRequest, PredictionResponse
from app.services.preprocessing_service import build_forecast_row, build_halt_row
from app.services.forecasting_service import predict_sales
from app.services.halt_service import predict_halt
from app.utils.feature_engineering import compute_rolling_stats

import joblib, os

logger  = logging.getLogger(__name__)
router  = APIRouter(prefix="/predict", tags=["prediction"])
_MODELS = os.path.join(os.path.dirname(__file__), "..", "models")

_encoders = None


def _get_encoders():
    global _encoders
    if _encoders is None:
        path = os.path.join(_MODELS, "label_encoders.pkl")
        if os.path.exists(path):
            _encoders = joblib.load(path)
    return _encoders


@router.post("", response_model=PredictionResponse)
async def run_prediction(request: PredictRequest):
    """
    Predict sales for a *future* day that has no sales data yet.

    Flow:
    1. Accept store context + target date from the request body.
    2. Fetch the last 30 historical records for this store from MongoDB
       (all records strictly before the requested date) to compute
       rolling averages and std — the only thing DB is used for.
    3. Forecasting preprocessing -> XGBRegressor -> predicted_sales (>= 0).
    4. Halt feature engineering using predicted_sales + history stats.
    5. XGBClassifier -> Automated | Human Review + reason.
    """
    db    = get_db()
    store = request.store
    target_date = str(request.sale_date)   # ISO string e.g. "2024-08-01"

    # -- 1. Fetch 30 most-recent historical records BEFORE the target date -----
    # These are used exclusively to build rolling stats (avg_7, avg_30, std_7, std_30).
    # We require sale_date < target_date so we never contaminate the future window.
    cursor = (
        db["sale_records"]
        .find({"store": store, "sale_date": {"$lt": target_date}})
        .sort([("sale_date", -1), ("created_at", -1)])
        .limit(30)
    )
    history_docs = await cursor.to_list(length=30)

    if not history_docs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No historical records found for store {store} before {target_date}. "
                "Please upload historical data first via the bulk upload script, "
                "or add past records via POST /new."
            ),
        )

    # Reverse to oldest->newest for rolling computation
    history_docs = list(reversed(history_docs))

    # -- 2. Compute rolling stats from history ---------------------------------
    prior_sales: list[float] = [
        float(doc["sales"]) for doc in history_docs if "sales" in doc
    ]

    if not prior_sales:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Historical records for store {store} exist but contain no 'sales' field. "
                "Ensure historical data was uploaded correctly."
            ),
        )

    stats = compute_rolling_stats(prior_sales)
    avg_7  = stats.get("average_sales_7",  0.0)
    avg_30 = stats.get("average_sales_30", avg_7)
    std_7  = stats.get("demand_std_7",     0.0)
    std_30 = stats.get("demand_std_30",    std_7)

    logger.info(
        "Store %s | target=%s | history=%d records | avg_7=%.0f avg_30=%.0f",
        store, target_date, len(prior_sales), avg_7, avg_30,
    )

    # -- 3. Forecast preprocessing -> prediction --------------------------------
    # Build the feature row from the REQUEST (not a DB document) — this is
    # the future-day context the user supplied.
    record_dict = request.model_dump()
    record_dict["sale_date"] = target_date   # ensure string for pd.to_datetime

    forecast_df = build_forecast_row(
        record   = record_dict,
        avg_7    = avg_7,
        avg_30   = avg_30,
        std_7    = std_7,
        std_30   = std_30,
        encoders = _get_encoders(),
    )
    predicted_sales = predict_sales(forecast_df)   # always >= 0 (clamped in service)
    logger.info("Store %s | %s -> predicted_sales=%.0f", store, target_date, predicted_sales)

    # -- 4. Halt feature derivation ---------------------------------------------
    promo_sales_list  = [float(d["sales"]) for d in history_docs if d.get("promo") == 1 and "sales" in d]
    normal_sales_list = [float(d["sales"]) for d in history_docs if d.get("promo") == 0 and "sales" in d]
    past_promo_sales  = sum(promo_sales_list)  / max(len(promo_sales_list),  1)
    past_normal_sales = sum(normal_sales_list) / max(len(normal_sales_list), 1)

    # Count consecutive non-promo days at tail of history
    days_since_promo = 0
    for doc in reversed(history_docs):
        if doc.get("promo") == 1:
            break
        days_since_promo += 1

    volatility    = std_7 / (avg_7 + 1)
    fc_confidence = 1.0 / (1.0 + std_7)

    halt_df = build_halt_row(
        predicted_sales     = predicted_sales,
        avg_7               = avg_7,
        avg_30              = avg_30,
        std_7               = std_7,
        std_30              = std_30,
        promo               = request.promo,
        past_promo_sales    = past_promo_sales,
        past_normal_sales   = past_normal_sales,
        days_since_promo    = days_since_promo,
        forecast_confidence = fc_confidence,
    )

    # -- 5. Halt prediction ----------------------------------------------------
    decision, reason, confidence = predict_halt(
        feature_df = halt_df,
        avg_7      = avg_7,
        predicted  = predicted_sales,
        std_7      = std_7,
        volatility = volatility,
    )

    return PredictionResponse(
        store           = store,
        sale_date       = request.sale_date,
        predicted_sales = predicted_sales,
        halt_decision   = decision,
        reason          = reason,
        confidence      = confidence,
    )