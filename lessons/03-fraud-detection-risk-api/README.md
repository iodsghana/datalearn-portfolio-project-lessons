# Lesson 03: Build a Fraud/Risk Detection API With Home Credit Data

## What You Will Build

In this lesson, you will build a suspicious loan-application screening project.

By the end, you will have:

- Used real Home Credit loan application data
- Documented the difference between confirmed fraud and risk proxy labels
- Engineered application, bureau, and previous-application features
- Trained an Isolation Forest anomaly model
- Trained a supervised high-risk classifier
- Tuned a decision threshold with F2 score
- Built APPROVE, REVIEW, and DECLINE decision logic
- Built a FastAPI scoring service
- Added a simple explainability endpoint
- Written tests
- Prepared a GitHub-ready project

This project teaches fraud-style screening patterns, but it also teaches professional honesty: the dataset does not contain verified fraud labels.

## Important Label Note

Home Credit provides `TARGET`, where:

```text
TARGET = 1 means the borrower had payment difficulty/default risk
TARGET = 0 means the borrower repaid
```

This is not a confirmed fraud label.

For this lesson, we treat `TARGET=1` as a high-risk proxy. That means the project is best described as:

```text
Application fraud/risk screening
```

Do not claim:

```text
This model detects confirmed fraud.
```

Better:

```text
This model screens applications for suspicious or high-risk patterns using Home Credit payment difficulty as a proxy label.
```

## Dataset

Dataset: Home Credit Default Risk  
Source: https://www.kaggle.com/competitions/home-credit-default-risk/data

Files needed:

| File | Purpose |
| --- | --- |
| `application_train.csv` | Main application table and `TARGET` label |
| `bureau.csv` | External credit bureau history |
| `previous_application.csv` | Prior Home Credit applications |
| `HomeCredit_columns_description.csv` | Data dictionary |

Place them here:

```text
data/raw/application_train.csv
data/raw/bureau.csv
data/raw/previous_application.csv
data/raw/HomeCredit_columns_description.csv
```

## Business Problem

The business wants to route applications into three groups:

| Decision | Meaning |
| --- | --- |
| `APPROVE` | Low risk, continue normal workflow |
| `REVIEW` | Uncertain or suspicious, send to analyst |
| `DECLINE` | High risk, requires strict policy review |

The model does not make final lending decisions. It provides a screening signal.

## Architecture

```text
Raw Home Credit CSVs
  -> data validation
  -> application feature engineering
  -> bureau history aggregation
  -> previous-application aggregation
  -> Stage 1: Isolation Forest anomaly score
  -> Stage 2: supervised classifier
  -> FastAPI score + decision
```

## Before You Start

Create the project:

```bash
mkdir fraud-detection-risk-api
cd fraud-detection-risk-api
```

Create folders:

```text
fraud-detection-risk-api/
  api/
  data/
    raw/
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
data/raw/
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
joblib>=1.4,<2
matplotlib>=3.8,<4
fastapi>=0.111,<1
uvicorn[standard]>=0.30,<1
pydantic>=2.7,<3
pytest>=8,<9
httpx>=0.27,<1
```

Create and activate the environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Exercise

Run:

```bash
python -c "import pandas, sklearn, fastapi; print('fraud risk environment ready')"
```

### Checkpoint Question

Why should this project use the phrase "risk proxy" instead of "confirmed fraud"?

## Module 2: Validate Raw Data

### Explanation

A professional project checks that required files and columns exist before training.

Create `src/data_validation.py`:

```python
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MONITORING = ROOT / "monitoring"
MONITORING.mkdir(exist_ok=True)

REQUIRED_FILES = [
    "application_train.csv",
    "bureau.csv",
    "previous_application.csv",
    "HomeCredit_columns_description.csv",
]


class HomeCreditDataError(ValueError):
    pass


def validate_raw_files(raw_dir=RAW):
    missing_files = [name for name in REQUIRED_FILES if not (raw_dir / name).exists()]
    if missing_files:
        raise HomeCreditDataError(f"Missing required raw files: {missing_files}")

    app = pd.read_csv(raw_dir / "application_train.csv")
    bureau = pd.read_csv(raw_dir / "bureau.csv")
    previous = pd.read_csv(raw_dir / "previous_application.csv")

    required_app_cols = ["SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH"]
    missing_app_cols = [col for col in required_app_cols if col not in app.columns]

    report = {
        "files": {
            "application_train.csv": {"rows": int(len(app)), "columns": int(app.shape[1])},
            "bureau.csv": {"rows": int(len(bureau)), "columns": int(bureau.shape[1])},
            "previous_application.csv": {"rows": int(len(previous)), "columns": int(previous.shape[1])},
        },
        "target": {
            "event_rate": round(float(app["TARGET"].mean()), 5),
            "valid_values": sorted(app["TARGET"].dropna().unique().tolist()),
        },
        "missing_application_columns": missing_app_cols,
        "join_coverage": {
            "bureau": round(float(app["SK_ID_CURR"].isin(bureau["SK_ID_CURR"]).mean()), 4),
            "previous_application": round(float(app["SK_ID_CURR"].isin(previous["SK_ID_CURR"]).mean()), 4),
        },
    }
    return report


def write_validation_report(raw_dir=RAW):
    report = validate_raw_files(raw_dir)
    out = MONITORING / "data_validation_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(write_validation_report())
```

Run:

```bash
python src/data_validation.py
```

### Practice Task

Add a check that confirms `TARGET` contains only `0` and `1`.

### Checkpoint Question

What is join coverage, and why does it matter?

## Module 3: Build Application Risk Features

Create `src/features.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "data" / "raw"

ID_COL = "SK_ID_CURR"
TARGET = "TARGET"


def safe_ratio(num, den):
    return num / den.replace({0: np.nan})


def add_application_features(df):
    df = df.copy()
    df["age_years"] = (-df["DAYS_BIRTH"] / 365.25).clip(18, 100)
    df["employment_years"] = (
        -df["DAYS_EMPLOYED"].replace(365243, np.nan) / 365.25
    ).clip(0, 60)
    df["registration_years"] = (-df["DAYS_REGISTRATION"] / 365.25).clip(0, 80)
    df["id_age_years"] = (-df["DAYS_ID_PUBLISH"] / 365.25).clip(0, 30)
    df["credit_to_income_ratio"] = safe_ratio(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["annuity_to_income_ratio"] = safe_ratio(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["goods_to_credit_ratio"] = safe_ratio(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
    df["external_score_mean"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["external_score_min"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis=1)
    df["social_default_rate_30"] = safe_ratio(
        df["DEF_30_CNT_SOCIAL_CIRCLE"],
        df["OBS_30_CNT_SOCIAL_CIRCLE"],
    )
    return df
```

### Simple Explanation

Fraud and risk screening often looks for unusual combinations:

- Very high credit compared with income
- Short employment history
- Weak external scores
- Social circle defaults
- Recently issued identity documents

### Exercise

Add this feature:

```text
credit_per_child
```

Hint:

```python
df["credit_per_child"] = df["AMT_CREDIT"] / (df["CNT_CHILDREN"] + 1)
```

## Module 4: Aggregate Bureau Features

### Explanation

The bureau table contains many credit records per applicant. We need one feature row per applicant.

Add:

```python
def build_bureau_features(raw_dir=RAW_DATA_DIR):
    bureau = pd.read_csv(raw_dir / "bureau.csv")
    bureau["is_active"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)
    bureau["is_closed"] = (bureau["CREDIT_ACTIVE"] == "Closed").astype(int)

    grouped = bureau.groupby(ID_COL).agg(
        bureau_credit_count=("SK_ID_BUREAU", "count"),
        bureau_active_count=("is_active", "sum"),
        bureau_closed_count=("is_closed", "sum"),
        bureau_overdue_days_max=("CREDIT_DAY_OVERDUE", "max"),
        bureau_overdue_days_sum=("CREDIT_DAY_OVERDUE", "sum"),
        bureau_total_credit=("AMT_CREDIT_SUM", "sum"),
        bureau_total_debt=("AMT_CREDIT_SUM_DEBT", "sum"),
        bureau_total_overdue=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        bureau_prolongations=("CNT_CREDIT_PROLONG", "sum"),
        bureau_recent_credit_days_mean=("DAYS_CREDIT", "mean"),
    ).reset_index()

    grouped["bureau_debt_to_credit_ratio"] = safe_ratio(
        grouped["bureau_total_debt"],
        grouped["bureau_total_credit"],
    )
    return grouped
```

### Practice Task

Add:

```text
bureau_overdue_per_account
```

### Checkpoint Question

Why can overdue bureau debt be useful for suspicious application screening?

## Module 5: Aggregate Previous Applications

Add:

```python
def build_previous_application_features(raw_dir=RAW_DATA_DIR):
    previous = pd.read_csv(raw_dir / "previous_application.csv")
    previous["was_approved"] = (previous["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    previous["was_refused"] = (previous["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    previous["was_canceled"] = (previous["NAME_CONTRACT_STATUS"] == "Canceled").astype(int)

    grouped = previous.groupby(ID_COL).agg(
        prev_application_count=("SK_ID_PREV", "count"),
        prev_approved_count=("was_approved", "sum"),
        prev_refused_count=("was_refused", "sum"),
        prev_canceled_count=("was_canceled", "sum"),
        prev_credit_total=("AMT_CREDIT", "sum"),
        prev_application_total=("AMT_APPLICATION", "sum"),
        prev_annuity_mean=("AMT_ANNUITY", "mean"),
        prev_down_payment_mean=("AMT_DOWN_PAYMENT", "mean"),
        prev_decision_days_mean=("DAYS_DECISION", "mean"),
    ).reset_index()

    grouped["prev_refusal_rate"] = safe_ratio(
        grouped["prev_refused_count"],
        grouped["prev_application_count"],
    )
    return grouped
```

### Exercise

Why might an applicant with several prior refused applications need review?

## Module 6: Create the Modeling Dataset

Add:

```python
APPLICATION_COLUMNS = [
    ID_COL,
    TARGET,
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
]


def load_modeling_data(raw_dir=RAW_DATA_DIR):
    app = pd.read_csv(raw_dir / "application_train.csv", usecols=lambda c: c in APPLICATION_COLUMNS)
    df = add_application_features(app)
    df = df.merge(build_bureau_features(raw_dir), on=ID_COL, how="left")
    df = df.merge(build_previous_application_features(raw_dir), on=ID_COL, how="left")
    df = df.replace([np.inf, -np.inf], np.nan)

    y = df.pop(TARGET).astype(int)
    X = df.drop(columns=[ID_COL])
    return X, y
```

### Practice Task

Run:

```python
from features import load_modeling_data
X, y = load_modeling_data()
print(X.shape)
print(y.mean())
```

### Checkpoint Question

Why should `SK_ID_CURR` not be used as a model feature?

## Module 7: Build the Preprocessor

### Explanation

The model needs a numeric matrix. Some columns are numbers, and others are categories.

Create `src/train.py`:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_cols),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
            ]), categorical_cols),
        ]
    )
```

### Exercise

Explain why missing values should be handled inside the pipeline instead of manually filling them once.

## Module 8: Train Stage 1 Anomaly Detection

### Explanation

Isolation Forest learns unusual patterns. In this project, we fit it on `TARGET=0` rows so it learns what normal applications look like.

Add:

```python
import numpy as np
from sklearn.ensemble import IsolationForest


def fit_anomaly_model(preprocessor, X_train, y_train, X_test):
    X_train_matrix = preprocessor.fit_transform(X_train)
    X_test_matrix = preprocessor.transform(X_test)

    iso = IsolationForest(
        n_estimators=250,
        contamination=max(float(y_train.mean()), 0.01),
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train_matrix[y_train.to_numpy() == 0])

    train_anomaly = -iso.score_samples(X_train_matrix)
    test_anomaly = -iso.score_samples(X_test_matrix)
    return iso, train_anomaly, test_anomaly
```

### Checkpoint Question

Why fit the anomaly model on `TARGET=0` rows only?

## Module 9: Train Stage 2 Supervised Classifier

### Explanation

Now we add the anomaly score as a new feature and train a supervised model.

Add:

```python
import joblib
import json
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from features import load_modeling_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MONITORING = ROOT / "monitoring"
MODEL_DIR.mkdir(exist_ok=True)
MONITORING.mkdir(exist_ok=True)


def train():
    X, y = load_modeling_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor(X_train)
    iso, train_anomaly, test_anomaly = fit_anomaly_model(preprocessor, X_train, y_train, X_test)

    X_train_enriched = X_train.assign(anomaly_score=train_anomaly)
    X_test_enriched = X_test.assign(anomaly_score=test_anomaly)

    classifier = Pipeline([
        ("prep", build_preprocessor(X_train_enriched)),
        ("model", HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    classifier.fit(X_train_enriched, y_train)
    prob = classifier.predict_proba(X_test_enriched)[:, 1]

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, prob), 4),
        "avg_precision": round(average_precision_score(y_test, prob), 4),
        "brier_score": round(brier_score_loss(y_test, prob), 4),
    }

    artifact = {
        "raw_preprocessor": preprocessor,
        "isolation_forest": iso,
        "classifier": classifier,
        "features": list(X.columns),
        "threshold": 0.20,
    }
    joblib.dump(artifact, MODEL_DIR / "fraud_model.pkl")
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)


if __name__ == "__main__":
    train()
```

For a faster smoke run, sample the dataset before splitting:

```python
sample = X.assign(_target=y).sample(20000, random_state=42)
y = sample.pop("_target")
X = sample
```

### Exercise

Train once with 20,000 rows. Record ROC-AUC and average precision.

## Module 10: Tune an F2 Threshold

### Explanation

Fraud/risk screening often cares more about recall than precision. F2 score weights recall more heavily.

Add:

```python
from sklearn.metrics import precision_recall_curve


def best_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f2 = (5 * precision * recall) / (4 * precision + recall + 1e-8)
    index = int(np.nanargmax(f2[:-1]))
    return float(thresholds[index]), float(f2[index])
```

### Practice Task

Print the best threshold and F2 score.

### Checkpoint Question

Why might a fraud team prefer F2 over F1?

## Module 11: Build Decision Logic

Create `src/predict.py`:

```python
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "fraud_model.pkl"
META_PATH = ROOT / "models" / "model_metadata.json"


class FraudModelRegistry:
    artifact = None
    metadata = None

    @classmethod
    def load(cls):
        if cls.artifact is None:
            cls.artifact = joblib.load(MODEL_PATH)
        if cls.metadata is None:
            cls.metadata = json.loads(META_PATH.read_text())
        return cls.artifact, cls.metadata


def decision(probability, threshold):
    if probability >= threshold:
        return "DECLINE"
    if probability >= threshold * 0.5:
        return "REVIEW"
    return "APPROVE"


def predict(record):
    artifact, meta = FraudModelRegistry.load()
    frame = pd.DataFrame([record]).reindex(columns=artifact["features"])
    matrix = artifact["raw_preprocessor"].transform(frame)
    anomaly_score = float(-artifact["isolation_forest"].score_samples(matrix)[0])
    enriched = frame.assign(anomaly_score=anomaly_score)
    probability = float(artifact["classifier"].predict_proba(enriched)[0, 1])
    threshold = artifact["threshold"]
    return {
        "fraud_risk_probability": round(probability, 4),
        "decision": decision(probability, threshold),
        "anomaly_score": round(anomaly_score, 4),
        "model_version": meta.get("sha256", "local")[:12],
    }
```

### Exercise

Change the logic so very high anomaly scores force `REVIEW` even if probability is low.

## Module 12: Build the FastAPI App

Create `api/app.py`:

```python
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict import FraudModelRegistry, predict

app = FastAPI(title="Home Credit Application Fraud/Risk API")


class Application(BaseModel):
    NAME_CONTRACT_TYPE: str = "Cash loans"
    CODE_GENDER: str = "F"
    FLAG_OWN_CAR: str = "N"
    FLAG_OWN_REALTY: str = "Y"
    CNT_CHILDREN: int = Field(0, ge=0)
    AMT_INCOME_TOTAL: float = Field(..., gt=0)
    AMT_CREDIT: float = Field(..., gt=0)
    AMT_ANNUITY: Optional[float] = Field(None, ge=0)
    AMT_GOODS_PRICE: Optional[float] = Field(None, ge=0)
    DAYS_BIRTH: int = Field(..., le=-1)
    DAYS_EMPLOYED: int
    DAYS_REGISTRATION: Optional[float] = None
    DAYS_ID_PUBLISH: Optional[float] = None
    EXT_SOURCE_1: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_2: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_3: Optional[float] = Field(None, ge=0, le=1)


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/readyz")
def ready():
    try:
        _, meta = FraudModelRegistry.load()
        return {"status": "ready", "model_version": meta.get("sha256", "local")[:12]}
    except Exception as exc:
        raise HTTPException(503, detail=str(exc))


@app.post("/v1/predict")
def score(application: Application):
    start = time.perf_counter()
    result = predict(application.model_dump())
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

### Practice Task

Add:

```text
POST /v1/predict/batch
```

It should accept multiple applications.

## Module 13: Add Explainability

### Explanation

A simple explanation endpoint can return important engineered values for review.

Add to `src/predict.py`:

```python
def explain(record):
    probability_result = predict(record)
    return {
        **probability_result,
        "key_inputs": {
            "AMT_INCOME_TOTAL": record.get("AMT_INCOME_TOTAL"),
            "AMT_CREDIT": record.get("AMT_CREDIT"),
            "EXT_SOURCE_2": record.get("EXT_SOURCE_2"),
            "EXT_SOURCE_3": record.get("EXT_SOURCE_3"),
        },
        "note": "This explanation is a lightweight review aid, not a causal explanation.",
    }
```

Add to the API:

```python
from predict import explain


@app.post("/v1/explain")
def explain_application(application: Application):
    return explain(application.model_dump())
```

### Exercise

Add `credit_to_income_ratio` to the explanation output.

## Module 14: Write Tests

Create `tests/test_fraud.py`:

```python
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "api"))

from features import add_application_features
from predict import decision


def payload():
    return {
        "AMT_INCOME_TOTAL": 162000,
        "AMT_CREDIT": 406597.5,
        "AMT_ANNUITY": 24700.5,
        "AMT_GOODS_PRICE": 351000,
        "DAYS_BIRTH": -12005,
        "DAYS_EMPLOYED": -4542,
        "DAYS_REGISTRATION": -3648,
        "DAYS_ID_PUBLISH": -2120,
        "EXT_SOURCE_1": 0.1,
        "EXT_SOURCE_2": 0.2,
        "EXT_SOURCE_3": 0.3,
        "OBS_30_CNT_SOCIAL_CIRCLE": 1,
        "DEF_30_CNT_SOCIAL_CIRCLE": 0,
    }


def test_application_features():
    result = add_application_features(pd.DataFrame([payload()]))
    assert "external_score_mean" in result.columns
    assert "credit_to_income_ratio" in result.columns
    assert result["age_years"].iloc[0] > 18


def test_decision_tiers():
    assert decision(0.01, 0.2) == "APPROVE"
    assert decision(0.15, 0.2) == "REVIEW"
    assert decision(0.25, 0.2) == "DECLINE"


def test_api_predict_with_mocked_model():
    from fastapi.testclient import TestClient
    import app as api_app

    mock_result = {
        "fraud_risk_probability": 0.12,
        "decision": "REVIEW",
        "anomaly_score": 0.34,
        "model_version": "abc123",
    }

    with patch("app.predict", return_value=mock_result), patch(
        "app.FraudModelRegistry.load",
        return_value=(MagicMock(), {"sha256": "abc123def456"}),
    ):
        client = TestClient(api_app.app)
        response = client.post("/v1/predict", json=payload())
        assert response.status_code == 200
        assert response.json()["decision"] == "REVIEW"
```

Run:

```bash
pytest tests/ -q
```

### Practice Task

Add a test that rejects a positive `DAYS_BIRTH`.

## Module 15: Write the Model Card

Create `monitoring/model_card.md`:

```md
# Model Card: Home Credit Application Fraud/Risk Screening

## Intended Use

Screen loan applications for elevated review risk.

## Label Note

`TARGET=1` is a payment difficulty/default risk label. It is used as a proxy for suspicious risk, not confirmed fraud.

## Model Design

Stage 1: Isolation Forest anomaly score  
Stage 2: supervised classifier using application, bureau, previous-application, and anomaly features

## Metrics

Report ROC-AUC, average precision, Brier score, threshold, and F2 score.

## Governance

Final decisions require compliance review, fairness review, policy rules, and human review.
```

### Exercise

Add your actual validation metrics after training.

## Module 16: Write the README

Your README should include:

- Project title
- Dataset source
- Important label note
- Business problem
- Architecture
- Quickstart
- API endpoints
- Metrics
- Governance notes
- Tests

### Practice Task

Write a short section called:

```text
Why This Is Not Confirmed Fraud Detection
```

## Final Assignment

Choose one extension:

1. Add a batch endpoint that summarizes decision counts.
2. Add an anomaly-only endpoint.
3. Add a fairness check by `CODE_GENDER`.
4. Add a monitoring chart for score distribution.
5. Add a rejected-application review queue export as CSV.

Your final assignment must include:

- Code
- One test
- README update
- One monitoring or explanation artifact

## Final Rubric

| Area | Excellent |
| --- | --- |
| Label honesty | Clearly explains `TARGET` is a proxy, not confirmed fraud |
| Data validation | Checks required files, target values, rows, and join coverage |
| Feature engineering | Uses application, bureau, and previous-application features |
| Anomaly model | Isolation Forest trained on normal applications |
| Supervised model | Classifier trained with anomaly score included |
| Thresholding | Uses F2 or another justified decision threshold |
| API | Provides health, readiness, predict, and explain endpoints |
| Tests | Covers features, decision logic, and API behavior |
| Governance | Discusses fairness, compliance, and human review |

## Interview Practice

Answer these aloud:

1. Why is `TARGET` not a fraud label?
2. Why use an anomaly model and a supervised model together?
3. Why train Isolation Forest on `TARGET=0` rows?
4. What is the difference between APPROVE, REVIEW, and DECLINE?
5. Why might F2 be more useful than accuracy?
6. What fairness concerns exist in fraud/risk screening?
7. How would you improve this project if you had confirmed fraud labels?

## Finished Project Reference

The finished project version is available here:

https://github.com/iodsghana/fraud-detection
