"""
SmartStock – FastAPI Application Entry Point
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.data import router as data_router
from app.routes.prediction import router as prediction_router
from app.database.mongodb import close_connection

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SmartStock API starting …")
    yield
    logger.info("SmartStock API shutting down …")
    await close_connection()


app = FastAPI(
    title       = "SmartStock API",
    description = "Retail demand forecasting & halt decisioning powered by XGBoost.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

app.include_router(data_router)
app.include_router(prediction_router)


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok", "service": "smartstock-api"}
