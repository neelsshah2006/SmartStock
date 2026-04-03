"""
SmartStock – Data Route
POST /new  →  validate, enrich, save to MongoDB
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from app.schemas.schemas import SaleRecordInput
from app.database.mongodb import get_db
from app.utils.validators import validate_sale_record
from app.utils.encoders import encode_value
from app.services.preprocessing_service import FORECAST_FEATURES

router = APIRouter(prefix="/new", tags=["data"])
logger = logging.getLogger(__name__)


@router.post("", status_code=status.HTTP_201_CREATED)
async def add_record(record: SaleRecordInput):
    """
    Accept new store/day data, run soft validation, and persist to MongoDB.

    The rolling stats (Average_Sales_7, etc.) are NOT stored here; they are
    computed on-the-fly at prediction time from the last N records in the DB.
    """
    warnings = validate_sale_record(record)
    if warnings:
        logger.warning("Soft validation warnings: %s", warnings)

    doc = record.model_dump()
    sale_date_str        = str(record.sale_date)   # e.g. "2023-07-15"
    doc["sale_date"]     = sale_date_str
    # Mirror upload_data.py: set created_at = sale_date so that
    # sort([("sale_date",-1),("created_at",-1)]) in prediction.py
    # ranks manually-added records correctly relative to bulk-uploaded data.
    # Records added in the future (sale_date > today) will still sort last
    # because ISO date strings sort lexicographically.
    doc["created_at"]    = sale_date_str

    db = get_db()
    result = await db["sale_records"].insert_one(doc)

    return {
        "status":   "saved",
        "id":       str(result.inserted_id),
        "warnings": warnings,
    }