"""
SmartStock – Data Upload Script
Reads train.csv + store.csv, applies the same cleaning from the notebook,
and bulk-inserts all records into MongoDB (collection: sale_records).

Usage:
    python upload_data.py
    python upload_data.py --train path/to/train.csv --store path/to/store.csv
    python upload_data.py --limit 10000   # upload only the first N rows (for testing)
"""

import argparse
import os
import sys
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from pymongo import MongoClient, InsertOne
from pymongo.errors import BulkWriteError
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MONGO_URI   = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME     = os.getenv("DB_NAME",   "smartstock")
COLLECTION  = "sale_records"
BATCH_SIZE  = 5_000   # records per bulk_write call


def load_and_clean(train_path: str, store_path: str) -> pd.DataFrame:
    logger.info("Loading CSVs …")
    train = pd.read_csv(train_path, low_memory=False)
    store = pd.read_csv(store_path)

    logger.info("  train shape: %s  |  store shape: %s", train.shape, store.shape)

    # ── Merge ────────────────────────────────────────────────────────────────
    data = train.merge(store, on="Store", how="left")
    data["StateHoliday"] = data["StateHoliday"].astype(str)

    # ── Drop unused columns ──────────────────────────────────────────────────
    DROP = [
        "Promo2", "Promo2SinceWeek", "Promo2SinceYear",
        "PromoInterval", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    ]
    data = data.drop(columns=DROP, errors="ignore")

    # ── Filter (mirror training logic) ───────────────────────────────────────
    data = data[data["Open"] == 1]
    data = data[data["Sales"] > 0]

    # ── Date features ────────────────────────────────────────────────────────
    data["Date"]       = pd.to_datetime(data["Date"])
    data["Year"]       = data["Date"].dt.year
    data["Month"]      = data["Date"].dt.month
    data["Day"]        = data["Date"].dt.day
    data["WeekOfYear"] = data["Date"].dt.isocalendar().week.astype(int)

    # ── Fill missing ─────────────────────────────────────────────────────────
    data["CompetitionDistance"] = data["CompetitionDistance"].fillna(
        data["CompetitionDistance"].median()
    )
    data = data.fillna(0)

    # ── Sort for rolling features ─────────────────────────────────────────────
    data = data.sort_values(["Store", "Date"]).reset_index(drop=True)

    # ── Rolling stats (store-level) ───────────────────────────────────────────
    logger.info("Computing rolling features …")
    for window, suffix in [(7, "7"), (30, "30")]:
        data[f"Average_Sales_{suffix}"] = (
            data.groupby("Store")["Sales"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        data[f"Demand_Std_{suffix}"] = (
            data.groupby("Store")["Sales"]
            .transform(lambda x: x.rolling(window, min_periods=2).std().fillna(0))
        )

    logger.info("Clean dataset shape: %s", data.shape)
    return data


def dataframe_to_docs(df: pd.DataFrame) -> list[dict]:
    """
    Convert each row to a MongoDB document dict.

    Field names MUST match exactly what prediction.py reads from the document:
        record["store"]                → "store"
        record["sale_date"]            → "sale_date"
        record["day_of_week"]          → "day_of_week"
        record["promo"]                → "promo"
        record["school_holiday"]       → "school_holiday"
        record["state_holiday"]        → "state_holiday"
        record["competition_distance"] → "competition_distance"
        record["store_type"]           → "store_type"
        record["assortment"]           → "assortment"
        doc["sales"]                   → "sales"   (used for rolling stats)
        doc["promo"]                   → "promo"   (used for promo history)
    """
    docs = []

    for row in df.itertuples(index=False):
        sale_date_str = str(row.Date.date())   # e.g. "2013-07-01"

        doc = {
            # ── Core fields – snake_case to match SaleRecordInput + prediction route ──
            "store":                int(row.Store),
            "sale_date":            sale_date_str,
            "day_of_week":          int(row.DayOfWeek),
            "promo":                int(row.Promo),
            "school_holiday":       int(row.SchoolHoliday),
            "state_holiday":        str(row.StateHoliday),
            "competition_distance": float(row.CompetitionDistance),
            "store_type":           str(row.StoreType),
            "assortment":           str(row.Assortment),
            # ── Actual sales (used by prediction route for rolling history) ──
            "sales":                float(row.Sales),
            # ── Date parts (informational) ───────────────────────────────────
            "year":                 int(row.Year),
            "month":                int(row.Month),
            "day":                  int(row.Day),
            "week_of_year":         int(row.WeekOfYear),
            # ── Pre-computed rolling features ─────────────────────────────────
            "average_sales_7":      float(row.Average_Sales_7),
            "demand_std_7":         float(row.Demand_Std_7),
            "average_sales_30":     float(row.Average_Sales_30),
            "demand_std_30":        float(row.Demand_Std_30),
            # ── Meta ──────────────────────────────────────────────────────────
            # IMPORTANT: created_at mirrors sale_date so that sort("created_at", -1)
            # in prediction.py correctly retrieves the chronologically latest record.
            # New records added via POST /new will have actual utcnow() timestamps
            # which are always > any historical sale_date, so they naturally sort last.
            "created_at":           sale_date_str,
            "source":               "bulk_upload",
        }
        docs.append(doc)
    return docs


def upload(docs: list[dict], mongo_uri: str, db_name: str) -> None:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5_000)

    # Quick connectivity check
    try:
        client.admin.command("ping")
        logger.info("Connected to MongoDB at %s", mongo_uri)
    except Exception as exc:
        logger.error("Cannot reach MongoDB: %s", exc)
        sys.exit(1)

    col = client[db_name][COLLECTION]

    total    = len(docs)
    inserted = 0

    for start in range(0, total, BATCH_SIZE):
        batch = docs[start : start + BATCH_SIZE]
        ops   = [InsertOne(d) for d in batch]

        try:
            result = col.bulk_write(ops, ordered=False)
            inserted += result.inserted_count
        except BulkWriteError as bwe:
            inserted += bwe.details.get("nInserted", 0)
            logger.warning(
                "Batch %d–%d had %d write error(s) (likely duplicates – safe to ignore).",
                start, start + len(batch),
                len(bwe.details.get("writeErrors", [])),
            )

        pct = inserted / total * 100
        logger.info("  Uploaded %d / %d records (%.1f%%)", inserted, total, pct)

    logger.info("✅ Done. %d records in '%s.%s'.", inserted, db_name, COLLECTION)
    client.close()


def main():
    parser = argparse.ArgumentParser(description="Upload Rossmann data to MongoDB")
    parser.add_argument("--train", default="data/train.csv",  help="Path to train.csv")
    parser.add_argument("--store", default="data/store.csv",  help="Path to store.csv")
    parser.add_argument("--limit", type=int, default=None,    help="Max rows to upload (for testing)")
    parser.add_argument("--uri",   default=MONGO_URI,         help="MongoDB URI")
    parser.add_argument("--db",    default=DB_NAME,           help="Database name")
    args = parser.parse_args()

    for path in [args.train, args.store]:
        if not os.path.exists(path):
            logger.error("File not found: %s", path)
            sys.exit(1)

    df = load_and_clean(args.train, args.store)

    if args.limit:
        df = df.head(args.limit)
        logger.info("Limiting upload to first %d rows.", args.limit)

    logger.info("Converting %d rows to documents …", len(df))
    docs = dataframe_to_docs(df)

    logger.info("Uploading to %s / %s.%s …", args.uri, args.db, COLLECTION)
    upload(docs, args.uri, args.db)


if __name__ == "__main__":
    main()
