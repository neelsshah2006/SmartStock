"""
SmartStock - Model Training Script
Trains the XGBRegressor (sales forecasting) and XGBClassifier (halt decisioning)
from the Rossmann dataset and serialises all artifacts to app/models/.
"""

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, classification_report, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "app", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 1. Load & Merge ──────────────────────────────────────────────────────────
print("[1/6] Loading data …")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), low_memory=False)
store = pd.read_csv(os.path.join(DATA_DIR, "store.csv"))
data  = train.merge(store, on="Store", how="left")
data["StateHoliday"] = data["StateHoliday"].astype(str)

# ── 2. Clean & Feature Engineer ──────────────────────────────────────────────
print("[2/6] Engineering features …")
DROP_COLS = [
    "Promo2", "Promo2SinceWeek", "Promo2SinceYear",
    "PromoInterval", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
]
data = data.drop(columns=DROP_COLS, errors="ignore")
data = data[data["Open"] == 1]
data = data[data["Sales"] > 0]

data["Date"]       = pd.to_datetime(data["Date"])
data["Year"]       = data["Date"].dt.year
data["Month"]      = data["Date"].dt.month
data["Day"]        = data["Date"].dt.day
data["WeekOfYear"] = data["Date"].dt.isocalendar().week.astype(int)
data["CompetitionDistance"] = data["CompetitionDistance"].fillna(data["CompetitionDistance"].median())
data = data.fillna(0)

# ── 3. Label Encoding ────────────────────────────────────────────────────────
print("[3/6] Encoding categoricals …")
label_encoders: dict[str, LabelEncoder] = {}
for col in ["StoreType", "Assortment", "StateHoliday"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# ── 4. Rolling Features ──────────────────────────────────────────────────────
data = data.sort_values(["Store", "Date"])
for window, suffix in [(7, "7"), (30, "30")]:
    data[f"Average_Sales_{suffix}"] = (
        data.groupby("Store")["Sales"]
        .transform(lambda x: x.rolling(window).mean())
    )
    data[f"Demand_Std_{suffix}"] = (
        data.groupby("Store")["Sales"]
        .transform(lambda x: x.rolling(window).std())
    )
data = data.fillna(0)

# ── 5. Train Forecasting Model ───────────────────────────────────────────────
print("[4/6] Training forecasting model (XGBRegressor) …")
FORECAST_FEATURES = [
    "Store", "DayOfWeek", "Promo", "SchoolHoliday",
    "StateHoliday", "Year", "Month", "Day", "WeekOfYear",
    "CompetitionDistance", "StoreType", "Assortment",
    "Average_Sales_7", "Average_Sales_30", "Demand_Std_7", "Demand_Std_30",
]
X = data[FORECAST_FEATURES]
y = data["Sales"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, shuffle=False)

forecast_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
forecast_model.fit(X_train, y_train)
y_pred = forecast_model.predict(X_val)
print(f"   Forecast MAPE: {mean_absolute_percentage_error(y_val, y_pred):.4f}")
print(f"   Sample predictions: {y_pred[:10]}")
print(f"   Min prediction: {y_pred.min()}, Max prediction: {y_pred.max()}")

# ── 6. Build Halt Features & Train Halt Model ────────────────────────────────
print("[5/6] Building halt features …")
val_df           = data.iloc[-len(X_val):].copy()
val_df["Actual"]    = y_val.values
val_df["Predicted"] = np.ceil(y_pred / 10) * 10

val_df["Error"]   = val_df["Actual"] - val_df["Predicted"]
val_df["APE"]     = abs(val_df["Error"]) / (val_df["Actual"] + 1)
val_df["Rolling_MAPE"] = (
    val_df.groupby("Store")["APE"].transform(lambda x: x.rolling(7).mean())
)
val_df["Error_Std"] = (
    val_df.groupby("Store")["Error"].transform(lambda x: x.rolling(7).std())
)
val_df["Forecast_Confidence"]  = 1 / (1 + val_df["Error_Std"])
val_df["Forecast_Uncertainty"] = val_df["Demand_Std_7"] / (val_df["Predicted"] + 1)

# Promo features
val_df["Past_Promo_Sales"] = (
    val_df["Actual"].where(val_df["Promo"] == 1)
    .groupby(val_df["Store"]).transform(lambda x: x.shift(1).expanding().mean())
)
val_df["Past_Normal_Sales"] = (
    val_df["Actual"].where(val_df["Promo"] == 0)
    .groupby(val_df["Store"]).transform(lambda x: x.shift(1).expanding().mean())
)
val_df["Expected_Uplift"]      = (val_df["Past_Promo_Sales"] - val_df["Past_Normal_Sales"]) / (val_df["Past_Normal_Sales"] + 1)
val_df["Current_Uplift"]       = (val_df["Actual"] - val_df["Average_Sales_7"]) / (val_df["Average_Sales_7"] + 1)
val_df["Promotion_Abnormality"] = val_df["Current_Uplift"] - val_df["Expected_Uplift"]

val_df["Volatility_Ratio"] = (val_df["Demand_Std_7"] / val_df["Average_Sales_7"]).replace([np.inf, -np.inf], 0).fillna(0)
val_df["Promo_Risk"]        = val_df["Promo"] * val_df["Volatility_Ratio"]

val_df["Days_Since_Last_Promo"] = (
    val_df.groupby("Store")["Promo"]
    .apply(lambda x: (~x.astype(bool)).cumsum() - (~x.astype(bool)).cumsum().where(x == 1).ffill().fillna(0))
    .reset_index(level=0, drop=True)
)
val_df["Volatility_Change"]  = (val_df["Demand_Std_7"] / val_df["Demand_Std_30"]).replace([np.inf, -np.inf], 0).fillna(0)
val_df["Trend_Shift"]        = val_df["Average_Sales_7"] - val_df["Average_Sales_30"]
val_df["Demand_Shock_Score"] = (val_df["Actual"] - val_df["Average_Sales_30"]) / (val_df["Demand_Std_30"] + 1)
val_df["Demand_Momentum"]    = (val_df["Average_Sales_7"] - val_df["Average_Sales_30"]) / (val_df["Average_Sales_30"] + 1)
val_df["Demand_Spike"]       = (val_df["Actual"] > val_df["Average_Sales_30"] + 2 * val_df["Demand_Std_30"]).astype(int)
val_df["Demand_Drop"]        = (val_df["Actual"] < val_df["Average_Sales_30"] - 2 * val_df["Demand_Std_30"]).astype(int)
val_df["Promo_Demand_Risk"]  = val_df["Promo"] * val_df["Volatility_Ratio"]
val_df["Forecast_Risk"]      = (1 - val_df["Forecast_Confidence"]) * val_df["Volatility_Ratio"]
val_df["HALT_Label"]         = (abs(val_df["Predicted"] - val_df["Actual"]) > val_df["Demand_Std_7"]).astype(int)
val_df = val_df.fillna(0)

HALT_FEATURES = [
    "Forecast_Confidence", "Forecast_Uncertainty", "Promotion_Abnormality",
    "Promo_Risk", "Past_Promo_Sales", "Past_Normal_Sales", "Days_Since_Last_Promo",
    "Volatility_Ratio", "Volatility_Change", "Trend_Shift", "Demand_Shock_Score",
    "Demand_Spike", "Demand_Drop", "Demand_Momentum", "Promo_Demand_Risk",
    "Forecast_Risk", "Predicted",
]
X_halt = val_df[HALT_FEATURES]
y_halt = val_df["HALT_Label"]

Xh_train, Xh_test, yh_train, yh_test = train_test_split(
    X_halt, y_halt, test_size=0.4, random_state=42, stratify=y_halt
)
scale_pos_weight = (len(yh_train) - sum(yh_train)) / sum(yh_train)

print("[6/6] Training halt model (XGBClassifier) …")
halt_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.01,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc", random_state=42,
)
halt_model.fit(Xh_train, yh_train)
yh_prob = halt_model.predict_proba(Xh_test)[:, 1]
yh_pred = halt_model.predict(Xh_test)
print(classification_report(yh_test, yh_pred))
print(f"   Halt ROC-AUC: {roc_auc_score(yh_test, yh_prob):.4f}")

# ── 7. Persist Artifacts ─────────────────────────────────────────────────────
joblib.dump(forecast_model,  os.path.join(MODELS_DIR, "forecasting_model.pkl"))
joblib.dump(halt_model,      os.path.join(MODELS_DIR, "halt_model.pkl"))
joblib.dump(label_encoders,  os.path.join(MODELS_DIR, "label_encoders.pkl"))
joblib.dump(FORECAST_FEATURES, os.path.join(MODELS_DIR, "forecast_features.pkl"))
joblib.dump(HALT_FEATURES,     os.path.join(MODELS_DIR, "halt_features.pkl"))

print("\n✅ All artifacts saved to app/models/")
print("   forecasting_model.pkl")
print("   halt_model.pkl")
print("   label_encoders.pkl")
print("   forecast_features.pkl")
print("   halt_features.pkl")
