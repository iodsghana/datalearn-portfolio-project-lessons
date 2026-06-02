# Lesson 06: Build an NYC Taxi Lakehouse and Demand Forecasting API

## What You Will Build

In this lesson, you will build an end-to-end data engineering and machine learning project using official NYC Taxi and Limousine Commission data.

By the end, you will have:

- Downloaded real NYC TLC taxi trip data
- Downloaded taxi zone lookup data
- Enriched trips with historical weather data
- Built a lakehouse-style folder structure
- Cleaned raw trip records
- Created a processed taxi trip fact table
- Created analytics marts for hourly demand and daily KPIs
- Validated raw and processed data quality
- Trained a zone-level hourly demand forecasting model
- Added anomaly detection for unusual demand/revenue patterns
- Built a FastAPI service for top zones and demand forecasts
- Written tests
- Created a model card and dashboard artifact

This project teaches both data engineering and machine learning.

## Real Data Sources

| Source | Use |
| --- | --- |
| NYC TLC Green Taxi Trip Records | Official trip-level taxi data |
| NYC TLC Taxi Zone Lookup | Zone names and borough mappings |
| Open-Meteo Historical Weather API | Hourly temperature, precipitation, and wind |

Useful links:

- NYC TLC Trip Record Data: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- NYC TLC parquet file pattern: `https://d37ci6vzurychx.cloudfront.net/trip-data/`
- Open-Meteo Archive API: https://open-meteo.com/en/docs/historical-weather-api

## Business Problem

A city operations team, taxi fleet operator, or mobility analytics team wants to know:

1. Which pickup zones have the most demand?
2. What is expected hourly demand for the next 24 to 72 hours?
3. Which zones or hours look unusual?
4. How do weather, hour of day, and rush hour affect trip demand?

## Lakehouse Concept

A lakehouse project usually separates data into layers:

```text
raw       -> original downloaded data
processed -> cleaned fact tables
marts     -> business-ready analytics tables
models    -> trained model artifacts
api       -> serving layer
```

This lesson uses:

```text
data/raw/
data/processed/
data/marts/
```

## Before You Start

Create the project:

```bash
mkdir nyc-taxi-lakehouse-demand
cd nyc-taxi-lakehouse-demand
```

Create folders:

```text
nyc-taxi-lakehouse-demand/
  api/
  data/
    raw/
    processed/
    marts/
  models/
  monitoring/
  src/
  tests/
```

Create `.gitignore`:

```txt
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
data/raw/*.parquet
data/raw/weather_hourly.csv
models/
monitoring/*.log
.env
```

## Module 1: Create the Environment

Create `requirements.txt`:

```txt
pandas>=2.2,<3
numpy>=1.26,<3
pyarrow>=15,<25
scikit-learn>=1.4,<2
joblib>=1.4,<2
matplotlib>=3.8,<4
fastapi>=0.111,<1
uvicorn[standard]>=0.30,<1
pydantic>=2.7,<3
pytest>=8,<9
httpx>=0.27,<1
```

Install:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Exercise

Run:

```bash
python -c "import pandas, pyarrow, sklearn, fastapi; print('taxi lakehouse ready')"
```

## Module 2: Download Official External Data

Create `src/download_data.py`:

```python
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

GREEN_TAXI_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "green_tripdata_2024-01.parquet"
)
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
WEATHER_API = "https://archive-api.open-meteo.com/v1/archive"


def download_file(url, path):
    if path.exists() and path.stat().st_size > 0:
        return path
    urllib.request.urlretrieve(url, path)
    return path


def download_weather():
    params = {
        "latitude": 40.7128,
        "longitude": -74.0060,
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "timezone": "America/New_York",
    }
    url = f"{WEATHER_API}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    hourly = payload["hourly"]
    df = pd.DataFrame({
        "pickup_hour": pd.to_datetime(hourly["time"]),
        "temperature_2m": hourly["temperature_2m"],
        "precipitation": hourly["precipitation"],
        "wind_speed_10m": hourly["wind_speed_10m"],
    })
    path = RAW_DIR / "weather_hourly.csv"
    df.to_csv(path, index=False)
    return path


def main():
    download_file(GREEN_TAXI_URL, RAW_DIR / "green_tripdata_2024-01.parquet")
    download_file(ZONE_LOOKUP_URL, RAW_DIR / "taxi_zone_lookup.csv")
    download_weather()
    print(f"Downloaded data to {RAW_DIR}")


if __name__ == "__main__":
    main()
```

Run:

```bash
python src/download_data.py
```

### Practice Task

List the files in `data/raw/`. You should see:

```text
green_tripdata_2024-01.parquet
taxi_zone_lookup.csv
weather_hourly.csv
```

### Checkpoint Question

Why is parquet a common format in modern data lake projects?

## Module 3: Validate Raw Data

Create `src/data_validation.py`:

```python
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MONITORING_DIR = ROOT / "monitoring"
MONITORING_DIR.mkdir(exist_ok=True)


def validate_raw_data():
    trips = pd.read_parquet(RAW_DIR / "green_tripdata_2024-01.parquet")
    trips["pickup_datetime"] = pd.to_datetime(trips["lpep_pickup_datetime"])
    trips["dropoff_datetime"] = pd.to_datetime(trips["lpep_dropoff_datetime"])
    duration_min = (trips["dropoff_datetime"] - trips["pickup_datetime"]).dt.total_seconds() / 60

    report = {
        "source": "NYC TLC Green Taxi Trip Records",
        "rows": int(len(trips)),
        "columns": int(trips.shape[1]),
        "pickup_min": trips["pickup_datetime"].min().isoformat(),
        "pickup_max": trips["pickup_datetime"].max().isoformat(),
        "valid_duration_rate": round(float(duration_min.between(1, 180).mean()), 4),
        "valid_distance_rate": round(float(trips["trip_distance"].between(0.1, 100).mean()), 4),
        "valid_amount_rate": round(float(trips["total_amount"].between(1, 500).mean()), 4),
    }
    out = MONITORING_DIR / "data_quality_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(validate_raw_data(), indent=2))
```

Run:

```bash
python src/data_validation.py
```

### Exercise

Add a check:

```text
PULocationID should not be missing
```

### Checkpoint Question

Why should we remove trips with impossible duration or distance?

## Module 4: Build the Processed Trip Fact Table

Create `src/etl.py`:

```python
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MART_DIR = ROOT / "data" / "marts"


def build_lakehouse():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MART_DIR.mkdir(parents=True, exist_ok=True)

    trips = pd.read_parquet(RAW_DIR / "green_tripdata_2024-01.parquet")
    zones = pd.read_csv(RAW_DIR / "taxi_zone_lookup.csv")

    trips["pickup_datetime"] = pd.to_datetime(trips["lpep_pickup_datetime"])
    trips["dropoff_datetime"] = pd.to_datetime(trips["lpep_dropoff_datetime"])
    trips["duration_min"] = (
        trips["dropoff_datetime"] - trips["pickup_datetime"]
    ).dt.total_seconds() / 60

    clean = trips[
        (trips["pickup_datetime"] >= "2024-01-01")
        & (trips["pickup_datetime"] < "2024-02-01")
        & (trips["duration_min"].between(1, 180))
        & (trips["trip_distance"].between(0.1, 100))
        & (trips["total_amount"].between(1, 500))
        & trips["PULocationID"].notna()
    ].copy()

    zone_cols = zones.rename(columns={
        "LocationID": "PULocationID",
        "Borough": "pickup_borough",
        "Zone": "pickup_zone",
        "service_zone": "pickup_service_zone",
    })
    clean = clean.merge(
        zone_cols[["PULocationID", "pickup_borough", "pickup_zone", "pickup_service_zone"]],
        on="PULocationID",
        how="left",
    )

    clean["pickup_hour"] = clean["pickup_datetime"].dt.floor("h")
    clean["pickup_date"] = clean["pickup_datetime"].dt.date
    clean.to_parquet(PROCESSED_DIR / "fact_green_taxi_trips.parquet", index=False)
    return clean
```

Run:

```bash
python -c "from src.etl import build_lakehouse; print(len(build_lakehouse()))"
```

### Practice Task

Open the processed parquet and answer:

1. How many cleaned trips remain?
2. Which pickup borough has the most trips?
3. Which pickup zone has the most trips?

## Module 5: Create the Hourly Zone Demand Mart

Add to `src/etl.py`:

```python
def build_hourly_mart(clean):
    hourly = clean.groupby(
        ["pickup_hour", "PULocationID", "pickup_zone", "pickup_borough"],
        dropna=False,
    ).agg(
        trips=("PULocationID", "size"),
        total_revenue=("total_amount", "sum"),
        avg_fare=("total_amount", "mean"),
        avg_distance=("trip_distance", "mean"),
        avg_duration_min=("duration_min", "mean"),
        avg_passengers=("passenger_count", "mean"),
    ).reset_index()

    hourly["hour"] = hourly["pickup_hour"].dt.hour
    hourly["dayofweek"] = hourly["pickup_hour"].dt.dayofweek
    hourly["is_weekend"] = hourly["dayofweek"].isin([5, 6]).astype(int)
    hourly["is_rush_hour"] = hourly["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    return hourly
```

Update `build_lakehouse()`:

```python
hourly = build_hourly_mart(clean)
hourly.to_csv(MART_DIR / "hourly_zone_demand.csv", index=False)
```

### Exercise

Add a column:

```text
revenue_per_trip
```

## Module 6: Add Weather Enrichment

Add:

```python
def load_weather():
    path = RAW_DIR / "weather_hourly.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["pickup_hour"])
```

Then merge weather into the hourly mart:

```python
weather = load_weather()
if not weather.empty:
    hourly = hourly.merge(weather, on="pickup_hour", how="left")

for col in ["temperature_2m", "precipitation", "wind_speed_10m"]:
    if col not in hourly.columns:
        hourly[col] = 0
    hourly[col] = hourly[col].ffill().bfill().fillna(0)
```

### Checkpoint Question

Why might weather help predict taxi demand?

## Module 7: Add Lag and Rolling Features

Add:

```python
hourly = hourly.sort_values(["PULocationID", "pickup_hour"])
hourly["lag_24h_trips"] = hourly.groupby("PULocationID")["trips"].shift(24)
hourly["rolling_24h_trips"] = (
    hourly.groupby("PULocationID")["trips"]
    .shift(1)
    .rolling(24, min_periods=1)
    .mean()
).reset_index(level=0, drop=True)

hourly["lag_24h_trips"] = hourly["lag_24h_trips"].fillna(hourly["trips"].median())
hourly["rolling_24h_trips"] = hourly["rolling_24h_trips"].fillna(hourly["trips"].median())
```

### Simple Explanation

`lag_24h_trips` answers:

```text
How many trips happened in this same zone 24 hours ago?
```

`rolling_24h_trips` answers:

```text
What was the recent average demand?
```

### Exercise

Add:

```text
lag_168h_trips
```

That is the same hour one week earlier.

## Module 8: Create Daily and Zone Summary Marts

Add:

```python
def build_daily_mart(clean):
    return clean.groupby(["pickup_date", "pickup_borough", "pickup_zone"], dropna=False).agg(
        trips=("PULocationID", "size"),
        revenue=("total_amount", "sum"),
        avg_fare=("total_amount", "mean"),
        avg_distance=("trip_distance", "mean"),
    ).reset_index()


def build_zone_summary(clean):
    return clean.groupby(["PULocationID", "pickup_zone", "pickup_borough"], dropna=False).agg(
        trips=("PULocationID", "size"),
        revenue=("total_amount", "sum"),
        avg_distance=("trip_distance", "mean"),
    ).reset_index().sort_values("trips", ascending=False)
```

Write them:

```python
daily.to_csv(MART_DIR / "daily_zone_kpis.csv", index=False)
zone_summary.to_csv(MART_DIR / "zone_summary.csv", index=False)
```

### Practice Task

Which table would a dashboard use for:

1. Top pickup zones?
2. Hourly demand forecast?
3. Daily revenue by zone?

## Module 9: Train the Demand Forecast Model

Create `src/train.py`:

```python
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
MART_DIR = ROOT / "data" / "marts"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES_NUM = [
    "hour", "dayofweek", "is_weekend", "is_rush_hour",
    "temperature_2m", "precipitation", "wind_speed_10m",
    "lag_24h_trips", "rolling_24h_trips",
]
FEATURES_CAT = ["pickup_zone", "pickup_borough"]
TARGET = "trips"


def build_model():
    preprocessor = ColumnTransformer([
        ("num", "passthrough", FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ])
    regressor = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("features", preprocessor), ("model", regressor)])


def train():
    hourly = pd.read_csv(MART_DIR / "hourly_zone_demand.csv", parse_dates=["pickup_hour"])

    top_zones = (
        hourly.groupby("pickup_zone")["trips"].sum()
        .sort_values(ascending=False)
        .head(12)
        .index
    )
    df = hourly[hourly["pickup_zone"].isin(top_zones)].copy()
    split_time = df["pickup_hour"].max() - pd.Timedelta(days=7)
    train_df = df[df["pickup_hour"] <= split_time]
    test_df = df[df["pickup_hour"] > split_time]

    model = build_model()
    model.fit(train_df[FEATURES_NUM + FEATURES_CAT], train_df[TARGET])
    preds = model.predict(test_df[FEATURES_NUM + FEATURES_CAT]).clip(min=0)

    rmse = float(np.sqrt(mean_squared_error(test_df[TARGET], preds)))
    mae = float(mean_absolute_error(test_df[TARGET], preds))

    artifact = {
        "forecast_model": model,
        "feature_num": FEATURES_NUM,
        "feature_cat": FEATURES_CAT,
        "top_zones": list(top_zones),
        "split_time": str(split_time),
    }
    joblib.dump(artifact, MODEL_DIR / "taxi_demand_model.pkl")

    metadata = {
        "source_data": "NYC TLC Green Taxi Trip Records, January 2024",
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "zones_modeled": int(len(top_zones)),
        "metrics": {"rmse": round(rmse, 4), "mae": round(mae, 4)},
        "top_zones": list(top_zones),
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    train()
```

Run:

```bash
python src/train.py
```

### Exercise

Record:

- MAE
- RMSE
- zones modeled

### Checkpoint Question

Why should the final 7 days be used as a time-based holdout?

## Module 10: Add Anomaly Detection

### Explanation

Anomaly detection flags unusual hours. For example:

- unusually high demand
- unusually low demand
- high revenue with low trip count
- strange fare patterns

Add to `src/train.py`:

```python
from sklearn.ensemble import IsolationForest

anomaly_features = df[[
    "trips", "total_revenue", "avg_fare", "avg_distance", "hour", "dayofweek"
]].fillna(0)

anomaly_model = IsolationForest(contamination=0.03, random_state=42)
anomaly_model.fit(anomaly_features)
df["is_anomaly"] = anomaly_model.predict(anomaly_features) == -1
```

Add `anomaly_model` to the saved artifact.

### Practice Task

Which zone has the most anomalous hours?

## Module 11: Create the Model Card

Create `monitoring/model_card.md`:

```md
# Model Card: NYC Taxi Demand Forecasting

## Intended Use

Forecast hourly taxi pickup demand by NYC pickup zone and flag abnormal demand patterns.

## Data

Official NYC TLC Green Taxi Trip Records, taxi zone lookup, and Open-Meteo historical weather.

## Evaluation

Report MAE and RMSE using the final 7 days as a holdout period.

## Governance Notes

This model supports operational planning. It should not be used for driver compensation, enforcement, or individual-level decisions without additional review.
```

### Exercise

Add your actual MAE and RMSE to the model card.

## Module 12: Build Prediction Helpers

Create `src/predict.py`:

```python
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "taxi_demand_model.pkl"
META_PATH = ROOT / "models" / "model_metadata.json"
MART_DIR = ROOT / "data" / "marts"


class DemandRegistry:
    artifact = None
    metadata = None

    @classmethod
    def load(cls):
        if cls.artifact is None:
            cls.artifact = joblib.load(MODEL_PATH)
        if cls.metadata is None:
            cls.metadata = json.loads(META_PATH.read_text())
        return cls.artifact, cls.metadata


def model_info():
    _, meta = DemandRegistry.load()
    return meta


def top_zones(limit=10):
    zone_summary = pd.read_csv(MART_DIR / "zone_summary.csv")
    return (
        zone_summary.head(limit)[["pickup_zone", "pickup_borough", "trips", "revenue"]]
        .rename(columns={"pickup_zone": "zone", "pickup_borough": "borough"})
        .to_dict(orient="records")
    )
```

### Practice Task

Add a function:

```text
zone_history(zone)
```

It should return recent hourly demand for one zone.

## Module 13: Forecast Future Hours

Add to `src/predict.py`:

```python
def forecast_zone(zone, horizon_hours=24):
    artifact, meta = DemandRegistry.load()
    hourly = pd.read_csv(MART_DIR / "hourly_zone_demand.csv", parse_dates=["pickup_hour"])
    history = hourly[hourly["pickup_zone"] == zone].sort_values("pickup_hour")
    if history.empty:
        raise ValueError(f"Unknown zone: {zone}")

    rows = []
    last = history.iloc[-1].copy()
    model = artifact["forecast_model"]
    features = artifact["feature_num"] + artifact["feature_cat"]

    for step in range(1, horizon_hours + 1):
        row = last.copy()
        row["pickup_hour"] = pd.to_datetime(last["pickup_hour"]) + pd.Timedelta(hours=step)
        row["hour"] = row["pickup_hour"].hour
        row["dayofweek"] = row["pickup_hour"].dayofweek
        row["is_weekend"] = int(row["dayofweek"] in [5, 6])
        row["is_rush_hour"] = int(row["hour"] in [7, 8, 9, 16, 17, 18])
        pred = float(model.predict(pd.DataFrame([row])[features])[0])
        rows.append({
            "pickup_hour": row["pickup_hour"].isoformat(),
            "predicted_trips": round(max(pred, 0), 2),
        })

    return {
        "zone": zone,
        "horizon_hours": horizon_hours,
        "model_version": meta.get("sha256", "local")[:12],
        "forecasts": rows,
    }
```

### Checkpoint Question

Why should forecasted trip counts never be negative?

## Module 14: Build the FastAPI App

Create `api/app.py`:

```python
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict import DemandRegistry, forecast_zone, model_info, top_zones

app = FastAPI(title="NYC Taxi Lakehouse and Demand Forecasting API")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    try:
        _, meta = DemandRegistry.load()
        return {"status": "ready", "model_version": meta.get("sha256", "local")[:12]}
    except Exception as exc:
        raise HTTPException(503, detail=str(exc))


@app.get("/v1/model/info")
def info():
    return model_info()


@app.get("/v1/zones/top")
def zones(limit: int = Query(10, ge=1, le=25)):
    return {"zones": top_zones(limit)}


@app.get("/v1/forecast")
def forecast(zone: str = "East Harlem North", horizon_hours: int = Query(24, ge=1, le=72)):
    start = time.perf_counter()
    try:
        result = forecast_zone(zone, horizon_hours)
    except Exception as exc:
        raise HTTPException(404, detail=str(exc))
    return {
        "request_id": str(uuid.uuid4()),
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        **result,
    }
```

Run:

```bash
uvicorn api.app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

### Exercise

Add:

```text
GET /v1/zones/history
```

It should return recent historical demand for a zone.

## Module 15: Write Tests

Create `tests/test_pipeline.py`:

```python
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "api"))

from data_validation import validate_raw_data
from etl import build_lakehouse
from predict import forecast_zone, model_info, top_zones


def test_raw_validation_report():
    report = validate_raw_data()
    assert report["rows"] > 0
    assert report["valid_duration_rate"] > 0.95


def test_lakehouse_marts_exist():
    result = build_lakehouse()
    assert result["clean_rows"] > 0
    hourly = pd.read_csv(ROOT / "data" / "marts" / "hourly_zone_demand.csv")
    assert {"pickup_hour", "pickup_zone", "trips"}.issubset(hourly.columns)


def test_model_info_and_forecast():
    info = model_info()
    assert info["metrics"]["mae"] >= 0
    zones = top_zones(3)
    assert len(zones) == 3
    forecast = forecast_zone(zones[0]["zone"], 6)
    assert len(forecast["forecasts"]) == 6
```

Run:

```bash
pytest tests/ -q
```

### Practice Task

Add an API test for:

```text
GET /v1/forecast?horizon_hours=0
```

It should fail validation.

## Module 16: Write the README

Your README should include:

- Project title
- Data sources
- Architecture
- Lakehouse layers
- Quickstart commands
- ETL outputs
- Model metrics
- API endpoints
- Tests
- Governance notes

### Practice Task

Write a section called:

```text
Why This Is a Lakehouse Project
```

Explain raw, processed, and mart layers in your own words.

## Final Assignment

Choose one extension:

1. Add Yellow Taxi data for the same month.
2. Add `lag_168h_trips` and compare model metrics.
3. Add a `GET /v1/anomalies` endpoint.
4. Add a daily demand forecast model.
5. Add a dashboard chart showing weather versus demand.

Your final assignment must include:

- Code
- One test
- README update
- One metric or chart

## Final Rubric

| Area | Excellent |
| --- | --- |
| Data ingestion | Downloads official NYC TLC and weather data |
| Data quality | Validates duration, distance, fare, and pickup zones |
| Lakehouse ETL | Produces raw, processed, and mart layers |
| Marts | Creates hourly demand, daily KPI, and zone summary tables |
| Feature engineering | Uses time, weather, lag, and rolling demand features |
| Forecasting | Uses time-based holdout and reports MAE/RMSE |
| Anomaly detection | Flags unusual demand/revenue hours |
| API | Serves model info, top zones, and forecasts |
| Tests | Covers validation, ETL, model, and API behavior |
| Governance | Explains operational use and limitations |

## Interview Practice

Answer these aloud:

1. What is the difference between raw, processed, and mart data?
2. Why is parquet useful for trip data?
3. What filters did you use to clean taxi trips?
4. Why is a time-based holdout better than random split here?
5. What features help predict taxi demand?
6. What does MAE mean in this project?
7. How would you scale this pipeline to all months of taxi data?
8. How would you monitor data quality in production?

## Finished Project Reference

The finished project version is available here:

https://github.com/iodsghana/NYC-Taxi-Lakehouse-and-Demand-Forecasting
