# Lesson 04: Build a Student Loan Default Forecasting API

## What You Will Build

In this lesson, you will build a forecasting and borrower-risk scoring project for student-loan default analytics.

By the end, you will have:

- Built a quarterly student-loan default-rate forecasting dataset
- Added macroeconomic signals such as unemployment, CPI, and interest rates
- Created backward-looking lag and rolling features
- Used chronological train, validation, and test splits
- Trained a portfolio-level default-rate forecaster
- Built stress scenarios such as recession, inflation, rate hike, and recovery
- Built borrower-level default-risk scoring features
- Served forecasts and borrower predictions with FastAPI
- Written tests
- Created a model card and dashboard-style monitoring artifact

The main lesson is not "how to make a perfect forecast." The main lesson is how to build a forecasting project without leaking future information.

## Real Data Sources

The finished project pattern can be built from these public data sources:

| Source | Use |
| --- | --- |
| Federal Student Aid Data Center | Student loan portfolio balances, repayment statuses, delinquency/default summaries |
| FRED Economic Data | Unemployment, CPI, federal funds rate, Treasury rates |
| College Scorecard | Institution-level completion, earnings, debt, and school type variables |

Helpful starting links:

- Federal Student Loan Portfolio: https://catalog-beta.data.gov/dataset/federal-student-loan-portfolio
- FRED: https://fred.stlouisfed.org/
- College Scorecard API: https://collegescorecard.ed.gov/data/api/

If a student does not have time to assemble all public data sources, they can start with a prepared demonstration CSV, then replace it with real data later. The modeling rules stay the same: chronological splits, no future leakage, and transparent scenario assumptions.

## Business Problem

A loan servicer or policy team wants to answer:

1. What will the default rate look like over the next few quarters?
2. How would default risk change under a recession or high-inflation scenario?
3. Which borrowers should receive early intervention support?

The API will support:

```text
GET  /v1/forecast
GET  /v1/scenarios
POST /v1/borrower/predict
```

## Key Forecasting Rule

Never train with information from the future.

Bad:

```text
Using 2024 unemployment to predict 2023 defaults
```

Good:

```text
Using lagged unemployment from earlier quarters to predict a future quarter
```

## Before You Start

Create the project:

```bash
mkdir student-loan-forecasting
cd student-loan-forecasting
```

Create folders:

```text
student-loan-forecasting/
  api/
  data/
  docs/
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
models/
monitoring/*.log
.env
```

## Module 1: Create the Python Environment

Create `requirements.txt`:

```txt
pandas>=2.2,<3
numpy>=1.26,<3
scikit-learn>=1.4,<2
xgboost>=2,<4
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
python -c "import pandas, sklearn, fastapi; print('forecast environment ready')"
```

## Module 2: Understand the Three Data Tables

You will use three CSVs:

```text
data/cohort_timeseries.csv
data/borrower_panel.csv
data/institutions.csv
```

### `cohort_timeseries.csv`

One row per quarter.

Important columns:

| Column | Meaning |
| --- | --- |
| `ds` | Quarter start date |
| `default_rate` | Portfolio default rate |
| `repayment_rate` | Share of borrowers in repayment |
| `deferment_rate` | Share in deferment |
| `unemployment_rate` | Macro stress signal |
| `cpi_pct` | Inflation signal |
| `fed_funds_rate` | Interest rate signal |
| `loan_volume_bn` | Loan volume |
| `avg_debt_at_grad` | Average debt at graduation |
| `median_income_6yr` | Earnings outcome signal |

### `borrower_panel.csv`

One row per borrower-quarter.

Important columns:

| Column | Meaning |
| --- | --- |
| `borrower_id` | Borrower identifier |
| `quarter` | Repayment quarter |
| `original_balance` | Starting balance |
| `current_balance` | Current balance |
| `annual_income` | Borrower income |
| `dti_ratio` | Debt-to-income ratio |
| `days_delinquent` | Days late |
| `loan_type` | Loan category |
| `institution_type` | School type |
| `repayment_plan` | Repayment plan |
| `default_flag` | Borrower default target |

### `institutions.csv`

Institution-level attributes from College Scorecard-style data.

### Practice Task

Open each CSV and write:

1. How many rows?
2. What is the target column?
3. Which columns are dates or quarters?

## Module 3: Write Data Validation

Create `src/data_validation.py`:

```python
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MONITORING = ROOT / "monitoring"
MONITORING.mkdir(exist_ok=True)


def validate():
    ts = pd.read_csv(DATA / "cohort_timeseries.csv", parse_dates=["ds"])
    borrower = pd.read_csv(DATA / "borrower_panel.csv")
    institutions = pd.read_csv(DATA / "institutions.csv")

    report = {
        "cohort_rows": int(len(ts)),
        "borrower_rows": int(len(borrower)),
        "institution_rows": int(len(institutions)),
        "date_min": ts["ds"].min().date().isoformat(),
        "date_max": ts["ds"].max().date().isoformat(),
        "default_rate_min": float(ts["default_rate"].min()),
        "default_rate_max": float(ts["default_rate"].max()),
        "borrower_default_rate": round(float(borrower["default_flag"].mean()), 5),
        "missing_values_top10": ts.isna().mean().sort_values(ascending=False).head(10).round(4).to_dict(),
    }
    out = MONITORING / "data_validation_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
```

Run:

```bash
python src/data_validation.py
```

### Exercise

Add a validation check:

```text
default_rate must be between 0 and 1
```

### Checkpoint Question

Why should time-series dates be sorted before modeling?

## Module 4: Build Time-Series Lag Features

### Explanation

Lag features use past values to predict future values.

Create `src/features.py`:

```python
import numpy as np
import pandas as pd


class TimeSeriesFeatureBuilder:
    LAG_PERIODS = [1, 2, 4, 8]
    ROLL_WINDOWS = [4, 8]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy().sort_values("ds").reset_index(drop=True)

        for lag in self.LAG_PERIODS:
            df[f"default_lag_{lag}q"] = df["default_rate"].shift(lag)
            df[f"repayment_lag_{lag}q"] = df["repayment_rate"].shift(lag)
            df[f"unemployment_lag_{lag}q"] = df["unemployment_rate"].shift(lag)

        for window in self.ROLL_WINDOWS:
            df[f"default_rollmean_{window}q"] = df["default_rate"].shift(1).rolling(window).mean()
            df[f"default_rollstd_{window}q"] = df["default_rate"].shift(1).rolling(window).std()

        df["default_qoq_change"] = df["default_rate"].diff(1)
        df["default_yoy_change"] = df["default_rate"].diff(4)
        df["quarter_of_year"] = pd.DatetimeIndex(df["ds"]).quarter
        df["q_sin"] = np.sin(2 * np.pi * df["quarter_of_year"] / 4)
        df["q_cos"] = np.cos(2 * np.pi * df["quarter_of_year"] / 4)

        return df.bfill().fillna(0)
```

### Simple Explanation

This line prevents leakage:

```python
df["default_rate"].shift(1).rolling(4).mean()
```

It says:

```text
Use the previous four quarters, not the current or future quarter.
```

### Exercise

Add:

```text
fed_rate_lag1q
cpi_lag2q
```

### Checkpoint Question

Why is `default_rate.rolling(4).mean()` risky without `shift(1)`?

## Module 5: Add Macro Regime Features

Add to `TimeSeriesFeatureBuilder.transform()`:

```python
df["is_recession"] = (df["unemployment_rate"] > 7.0).astype(int)
df["is_high_inflation"] = (df["cpi_pct"] > 5.0).astype(int)
df["income_to_debt"] = df["median_income_6yr"] / (df["avg_debt_at_grad"] + 1)
df["unemp_x_debt"] = df["unemployment_rate"] * df["avg_debt_at_grad"] / 10000
df["loan_vol_growth"] = df["loan_volume_bn"].pct_change(4)
```

### Practice Task

Create a table showing average default rate when:

```text
is_recession = 1
is_recession = 0
```

### Checkpoint Question

Why might unemployment affect student-loan repayment?

## Module 6: Create Forecast Metrics

Create `src/metrics.py` or add to `src/train.py`:

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(y_true, y_pred):
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    return float(np.mean(true_direction == pred_direction) * 100)


def eval_metrics(y_true, y_pred):
    return {
        "mape": round(mape(y_true, y_pred), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 6),
        "directional_acc": round(directional_accuracy(y_true, y_pred), 2),
    }
```

### Exercise

If actual default rate is `0.05` and predicted default rate is `0.055`, what is the percentage error?

Answer:

```text
10%
```

## Module 7: Split Chronologically

### Explanation

Forecasting projects should not shuffle rows. You must train on earlier dates and test on later dates.

Create `src/train.py`:

```python
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

TRAIN_END = "2019-12-31"
VAL_END = "2022-12-31"


def load_splits():
    df = pd.read_csv(DATA / "cohort_timeseries.csv", parse_dates=["ds"])
    df = df.sort_values("ds")

    train = df[df["ds"] <= TRAIN_END].copy()
    val = df[(df["ds"] > TRAIN_END) & (df["ds"] <= VAL_END)].copy()
    test = df[df["ds"] > VAL_END].copy()
    return train, val, test
```

### Practice Task

Print the number of quarters in each split.

### Checkpoint Question

What is wrong with using `train_test_split(..., shuffle=True)` for this problem?

## Module 8: Train a Portfolio Forecast Model

### Explanation

We will use a scikit-learn regression model as a practical forecaster. It learns from lagged default rates and macro signals.

Add:

```python
import joblib
import json
from sklearn.ensemble import GradientBoostingRegressor

from features import TimeSeriesFeatureBuilder
from metrics import eval_metrics

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def train_portfolio_model():
    train, val, test = load_splits()
    builder = TimeSeriesFeatureBuilder()

    train_feat = builder.fit(train).transform(train)
    val_feat = builder.transform(val)
    test_feat = builder.transform(test)

    drop_cols = ["ds", "default_rate"]
    X_train = train_feat.drop(columns=drop_cols)
    y_train = train_feat["default_rate"]
    X_test = test_feat.drop(columns=drop_cols)
    y_test = test_feat["default_rate"]

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = eval_metrics(y_test, pred)
    artifact = {
        "feature_builder": builder,
        "portfolio_model": model,
        "feature_columns": list(X_train.columns),
        "last_training_frame": train_feat.tail(8),
    }
    joblib.dump(artifact, MODEL_DIR / "forecast_model.pkl")
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)
```

Run:

```bash
python src/train.py
```

### Exercise

Record:

- MAPE
- RMSE
- Directional accuracy

## Module 9: Build Borrower-Level Features

### Explanation

The portfolio model forecasts total default rates. The borrower model scores individual borrower risk.

Add to `src/features.py`:

```python
class BorrowerFeatureBuilder:
    LOAN_TYPE_MAP = {
        "Direct Subsidized": 0,
        "Direct Unsubsidized": 1,
        "PLUS": 2,
        "Grad PLUS": 3,
    }
    INST_TYPE_MAP = {
        "public_4yr": 0,
        "private_nonprofit_4yr": 1,
        "for_profit": 2,
        "community_college": 3,
    }
    PLAN_MAP = {
        "Standard": 0,
        "Income-Driven": 1,
        "Graduated": 2,
        "Extended": 3,
        "SAVE": 4,
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["loan_type_enc"] = df["loan_type"].map(self.LOAN_TYPE_MAP).fillna(-1)
        df["inst_type_enc"] = df["institution_type"].map(self.INST_TYPE_MAP).fillna(-1)
        df["plan_enc"] = df["repayment_plan"].map(self.PLAN_MAP).fillna(-1)
        df["balance_to_income"] = df["current_balance"] / (df["annual_income"] + 1)
        df["balance_reduction"] = 1 - df["current_balance"] / (df["original_balance"] + 1)
        df["is_at_risk"] = (df["days_delinquent"] > 30).astype(int)
        df["is_severely_behind"] = (df["days_delinquent"] > 60).astype(int)
        df["macro_stress"] = df["unemployment_rate"] * df["cpi_pct"] / 10
        return df.drop(columns=["borrower_id", "loan_type", "institution_type", "repayment_plan"], errors="ignore").fillna(0)
```

### Exercise

Add:

```text
payment_progress = 1 - current_balance / original_balance
```

## Module 10: Train a Borrower Classifier

Add to `src/train.py`:

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from features import BorrowerFeatureBuilder


def train_borrower_model():
    df = pd.read_csv(DATA / "borrower_panel.csv")
    builder = BorrowerFeatureBuilder()
    X = builder.fit(df).transform(df.drop(columns=["default_flag"]))
    y = df["default_flag"]

    train_mask = df["quarter"] < 6
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, prob), 4),
        "avg_precision": round(average_precision_score(y_test, prob), 4),
    }
    return model, builder, metrics
```

### Checkpoint Question

Why split borrower records by quarter instead of random rows?

## Module 11: Add Stress Scenarios

Create `src/predict.py`:

```python
SCENARIOS = {
    "baseline": {
        "description": "Stable labor market and moderate inflation",
        "unemployment_rate": 4.5,
        "cpi_pct": 2.5,
        "fed_funds_rate": 4.0,
    },
    "recession": {
        "description": "Higher unemployment and weaker repayment conditions",
        "unemployment_rate": 8.0,
        "cpi_pct": 3.5,
        "fed_funds_rate": 2.5,
    },
    "high_inflation": {
        "description": "Elevated inflation pressure",
        "unemployment_rate": 5.0,
        "cpi_pct": 6.5,
        "fed_funds_rate": 5.5,
    },
    "rate_hike": {
        "description": "Higher interest rate environment",
        "unemployment_rate": 5.2,
        "cpi_pct": 4.0,
        "fed_funds_rate": 6.0,
    },
    "recovery": {
        "description": "Improving labor market and lower inflation",
        "unemployment_rate": 3.8,
        "cpi_pct": 2.0,
        "fed_funds_rate": 3.0,
    },
}
```

### Simple Explanation

A scenario is not a prediction by itself. It is an assumption.

Example:

```text
If unemployment rises to 8%, what might happen to default rates?
```

### Exercise

Create a new scenario called:

```text
soft_landing
```

## Module 12: Forecast Future Quarters

Create a model registry and forecast function:

```python
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "forecast_model.pkl"
META_PATH = ROOT / "models" / "model_metadata.json"


class ForecastRegistry:
    artifact = None
    metadata = None

    @classmethod
    def load(cls):
        if cls.artifact is None:
            cls.artifact = joblib.load(MODEL_PATH)
        if cls.metadata is None:
            cls.metadata = json.loads(META_PATH.read_text())
        return cls.artifact, cls.metadata


def forecast(horizon=4, scenario_name="baseline"):
    artifact, meta = ForecastRegistry.load()
    scenario = SCENARIOS[scenario_name]
    frame = artifact["last_training_frame"].copy()
    rows = []

    for step in range(horizon):
        next_date = pd.to_datetime(frame["ds"].iloc[-1]) + pd.offsets.QuarterBegin()
        new_row = frame.iloc[-1].copy()
        new_row["ds"] = next_date
        new_row["unemployment_rate"] = scenario["unemployment_rate"]
        new_row["cpi_pct"] = scenario["cpi_pct"]
        new_row["fed_funds_rate"] = scenario["fed_funds_rate"]

        expanded = pd.concat([frame, pd.DataFrame([new_row])], ignore_index=True)
        features = artifact["feature_builder"].transform(expanded).tail(1)
        X = features[artifact["feature_columns"]]
        pred = float(artifact["portfolio_model"].predict(X)[0])
        new_row["default_rate"] = pred
        frame = pd.concat([frame, pd.DataFrame([new_row])], ignore_index=True)

        rows.append({
            "quarter": str(next_date.date()),
            "forecast": round(pred, 4),
            "lower_90": round(max(pred - 0.01, 0), 4),
            "upper_90": round(min(pred + 0.01, 1), 4),
        })

    return {
        "scenario": scenario_name,
        "scenario_description": scenario["description"],
        "horizon_quarters": horizon,
        "model_version": meta.get("sha256", "local")[:12],
        "quarters": rows,
    }
```

### Checkpoint Question

Why do we append each predicted quarter before predicting the next one?

## Module 13: Predict Borrower Risk

Add:

```python
def risk_band(prob):
    if prob >= 0.30:
        return "Very High"
    if prob >= 0.15:
        return "High"
    if prob >= 0.07:
        return "Moderate"
    return "Low"


def predict_borrower(record):
    artifact, meta = ForecastRegistry.load()
    builder = artifact["borrower_feature_builder"]
    model = artifact["borrower_model"]
    frame = pd.DataFrame([record])
    X = builder.transform(frame)
    prob = float(model.predict_proba(X)[0, 1])
    return {
        "default_probability": round(prob, 4),
        "risk_band": risk_band(prob),
        "model_version": meta.get("sha256", "local")[:12],
    }
```

### Practice Task

Change the risk band thresholds and explain the business tradeoff.

## Module 14: Build the FastAPI App

Create `api/app.py`:

```python
import sys
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict import ForecastRegistry, SCENARIOS, forecast, predict_borrower

ScenarioName = Literal["baseline", "recession", "high_inflation", "rate_hike", "recovery"]

app = FastAPI(title="Student Loan Default Rate Forecasting API")


class BorrowerRequest(BaseModel):
    quarter: int = Field(..., ge=0, le=7)
    original_balance: float = Field(..., ge=1000)
    current_balance: float = Field(..., ge=0)
    annual_income: float = Field(..., ge=5000)
    dti_ratio: float = Field(..., ge=0, le=5)
    days_delinquent: int = Field(0, ge=0, le=365)
    unemployment_rate: float = Field(4.5, ge=0, le=25)
    cpi_pct: float = Field(2.5, ge=0, le=15)
    loan_type: str = "Direct Unsubsidized"
    institution_type: str = "public_4yr"
    repayment_plan: str = "Standard"


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/readyz")
def ready():
    try:
        _, meta = ForecastRegistry.load()
        return {"status": "ready", "sha": meta.get("sha256", "local")[:12]}
    except Exception as exc:
        raise HTTPException(503, detail=str(exc))


@app.get("/v1/forecast")
def get_forecast(
    horizon: int = Query(4, ge=1, le=12),
    scenario: ScenarioName = "baseline",
):
    start = time.perf_counter()
    result = forecast(horizon=horizon, scenario_name=scenario)
    return {
        "request_id": str(uuid.uuid4()),
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        **result,
    }


@app.post("/v1/borrower/predict")
def score_borrower(req: BorrowerRequest):
    start = time.perf_counter()
    result = predict_borrower(req.model_dump())
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

## Module 15: Add Scenario Comparison

Add to `src/predict.py`:

```python
def scenario_comparison(horizon=4):
    return {
        "horizon_quarters": horizon,
        "scenarios": {
            name: forecast(horizon=horizon, scenario_name=name)["quarters"]
            for name in SCENARIOS
        },
    }
```

Add to `api/app.py`:

```python
from predict import scenario_comparison


@app.get("/v1/scenarios")
def get_scenarios(horizon: int = Query(4, ge=1, le=12)):
    return scenario_comparison(horizon=horizon)
```

### Exercise

Which scenario produces the highest default forecast? Why?

## Module 16: Write Tests

Create `tests/test_forecast.py`:

```python
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "api"))

from features import BorrowerFeatureBuilder, TimeSeriesFeatureBuilder
from metrics import directional_accuracy, mape
from predict import SCENARIOS


def test_mape_perfect():
    y = np.array([0.05, 0.06, 0.04])
    assert mape(y, y) == 0.0


def test_scenarios_defined():
    assert "baseline" in SCENARIOS
    assert "recession" in SCENARIOS
    assert SCENARIOS["recession"]["unemployment_rate"] > SCENARIOS["baseline"]["unemployment_rate"]


def test_time_series_features_created():
    df = pd.DataFrame({
        "ds": pd.date_range("2014-01-01", periods=12, freq="QS"),
        "default_rate": np.linspace(0.03, 0.06, 12),
        "repayment_rate": np.linspace(0.80, 0.75, 12),
        "unemployment_rate": np.linspace(4, 8, 12),
        "cpi_pct": np.linspace(2, 5, 12),
        "fed_funds_rate": np.linspace(1, 4, 12),
        "loan_volume_bn": np.linspace(20, 30, 12),
        "avg_debt_at_grad": np.linspace(25000, 30000, 12),
        "median_income_6yr": np.linspace(40000, 50000, 12),
    })
    out = TimeSeriesFeatureBuilder().fit(df).transform(df)
    assert "default_lag_1q" in out.columns
    assert "q_sin" in out.columns


def test_borrower_features():
    row = pd.DataFrame([{
        "borrower_id": "B1",
        "quarter": 2,
        "original_balance": 28000,
        "current_balance": 26500,
        "annual_income": 38000,
        "dti_ratio": 0.74,
        "days_delinquent": 45,
        "unemployment_rate": 4.5,
        "cpi_pct": 2.5,
        "loan_type": "Direct Unsubsidized",
        "institution_type": "public_4yr",
        "repayment_plan": "Standard",
    }])
    out = BorrowerFeatureBuilder().fit(row).transform(row)
    assert "balance_to_income" in out.columns
    assert out["is_at_risk"].iloc[0] == 1
```

Run:

```bash
pytest tests/ -q
```

### Practice Task

Add an API test for invalid forecast horizon:

```text
/v1/forecast?horizon=13 should return 422
```

## Module 17: Write the Model Card

Create `monitoring/model_card.md`:

```md
# Model Card: Student Loan Default Forecasting

## Intended Use

Forecast quarterly student-loan default rates under macroeconomic scenarios and score borrower repayment risk for early intervention.

## Data

Student loan portfolio, macroeconomic, borrower, and institution-style features.

## Evaluation

Report MAPE, RMSE, MAE, and directional accuracy for the portfolio forecaster.

Report ROC-AUC and average precision for the borrower classifier.

## Governance Notes

Forecast scenarios are assumptions, not certainties.

Borrower scores should support early intervention and human review, not automated adverse action.
```

### Exercise

Add your actual metrics after training.

## Module 18: Write the README

Your README should include:

- Project title
- Business problem
- Data sources
- Forecasting approach
- Chronological split explanation
- Scenario descriptions
- API endpoints
- Metrics
- Governance notes
- Tests

### Practice Task

Write a section called:

```text
Why Chronological Validation Matters
```

## Final Assignment

Choose one extension:

1. Add a `soft_landing` macro scenario.
2. Add a monthly-to-quarterly FRED macro download script.
3. Add a forecast dashboard plot.
4. Add a borrower intervention recommendation such as `email`, `call`, or `financial counseling`.
5. Add a College Scorecard institution-risk feature.

Your final assignment must include:

- Code
- One test
- README update
- One metric or chart

## Final Rubric

| Area | Excellent |
| --- | --- |
| Data | Uses documented student loan, macro, and institution data sources |
| Validation | Checks ranges, dates, missingness, and target columns |
| Feature engineering | Uses lagged and rolling features without future leakage |
| Forecasting | Uses chronological splits and reports forecast metrics |
| Scenarios | Defines clear macro assumptions |
| Borrower scoring | Builds borrower-level features and risk bands |
| API | Serves forecast, scenarios, model info, and borrower scoring |
| Tests | Covers metrics, features, scenarios, and API behavior |
| Governance | Explains that forecasts are planning tools, not certainties |

## Interview Practice

Answer these aloud:

1. Why should time-series validation be chronological?
2. What is future leakage?
3. Why do we use lag features?
4. What is MAPE?
5. What does directional accuracy measure?
6. How are stress scenarios different from predictions?
7. How would you replace demo data with real FSA, FRED, and College Scorecard data?
8. How should borrower scores be used responsibly?

## Finished Project Reference

The finished project version is available here:

https://github.com/iodsghana/student-loan-forecasting
