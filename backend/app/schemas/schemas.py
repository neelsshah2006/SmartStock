"""
SmartStock – Pydantic v2 schemas
"""

from datetime import date
from pydantic import BaseModel, Field, field_validator


# ── Input (user submits new store/day record) ────────────────────────────────

class SaleRecordInput(BaseModel):
    """
    Historical record submitted via the Add New Data form.
    Includes actual sales — this is past/known data being stored so
    the prediction route can use it as rolling-stat context.
    Rolling / lag features (Average_Sales_7, etc.) are computed server-side.
    """
    store:                int   = Field(..., ge=1,  description="Store ID")
    sale_date:            date  = Field(...,         description="Date of the record")
    day_of_week:          int   = Field(..., ge=1, le=7, description="1=Mon … 7=Sun")
    sales:                float = Field(..., gt=0,  description="Actual sales for this day (known historical value)")
    promo:                int   = Field(..., ge=0, le=1,  description="1 if promotion active")
    school_holiday:       int   = Field(..., ge=0, le=1)
    state_holiday:        str   = Field(..., description="0 / a / b / c")
    competition_distance: float = Field(..., ge=0)
    store_type:           str   = Field(..., description="a / b / c / d")
    assortment:           str   = Field(..., description="a / b / c")

    @field_validator("state_holiday")
    @classmethod
    def validate_state_holiday(cls, v: str) -> str:
        allowed = {"0", "a", "b", "c"}
        if v not in allowed:
            raise ValueError(f"state_holiday must be one of {allowed}")
        return v

    @field_validator("store_type")
    @classmethod
    def validate_store_type(cls, v: str) -> str:
        if v not in {"a", "b", "c", "d"}:
            raise ValueError("store_type must be a, b, c, or d")
        return v

    @field_validator("assortment")
    @classmethod
    def validate_assortment(cls, v: str) -> str:
        if v not in {"a", "b", "c"}:
            raise ValueError("assortment must be a, b, or c")
        return v


# ── Prediction request (future day – no sales data yet) ─────────────────────

class PredictRequest(BaseModel):
    """
    Input for a *future* day prediction.
    The user supplies the store context and the date they want to predict.
    Historical sales are fetched from MongoDB automatically to build rolling features.
    No 'sales' field — that is what we are predicting.
    """
    store:                int   = Field(..., ge=1,  description="Store ID")
    sale_date:            date  = Field(...,         description="Future date to predict sales for")
    day_of_week:          int   = Field(..., ge=1, le=7, description="1=Mon … 7=Sun")
    promo:                int   = Field(..., ge=0, le=1,  description="1 if promotion active on this day")
    school_holiday:       int   = Field(..., ge=0, le=1)
    state_holiday:        str   = Field(..., description="0 / a / b / c")
    competition_distance: float = Field(..., ge=0)
    store_type:           str   = Field(..., description="a / b / c / d")
    assortment:           str   = Field(..., description="a / b / c")

    @field_validator("state_holiday")
    @classmethod
    def validate_state_holiday(cls, v: str) -> str:
        if v not in {"0", "a", "b", "c"}:
            raise ValueError("state_holiday must be one of: 0, a, b, c")
        return v

    @field_validator("store_type")
    @classmethod
    def validate_store_type(cls, v: str) -> str:
        if v not in {"a", "b", "c", "d"}:
            raise ValueError("store_type must be a, b, c, or d")
        return v

    @field_validator("assortment")
    @classmethod
    def validate_assortment(cls, v: str) -> str:
        if v not in {"a", "b", "c"}:
            raise ValueError("assortment must be a, b, or c")
        return v


# ── Prediction response ───────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    store:          int
    sale_date:      date
    predicted_sales: float
    halt_decision:  str   = Field(..., description="Automated | Human Review")
    reason:         str
    confidence:     float = Field(..., ge=0.0, le=1.0)


# ── DB document ──────────────────────────────────────────────────────────────

class SaleRecordDB(SaleRecordInput):
    """Stored document – enriched with computed rolling features."""
    year:              int | None = None
    month:             int | None = None
    day:               int | None = None
    week_of_year:      int | None = None
    average_sales_7:   float      = 0.0
    demand_std_7:      float      = 0.0
    average_sales_30:  float      = 0.0
    demand_std_30:     float      = 0.0
    # Encoded values cached here so prediction is fast
    state_holiday_enc: int        = 0
    store_type_enc:    int        = 0
    assortment_enc:    int        = 0