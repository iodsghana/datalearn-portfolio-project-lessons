# Lesson 02: Build a Credit Risk API With Real Home Credit Data

## What You Will Build

In this lesson, you will build an end-to-end credit risk machine learning project.

By the end, you will have:

- Downloaded real Home Credit loan application data
- Validated raw data quality
- Joined application, bureau, and previous-application tables
- Engineered credit risk features
- Trained and compared several classification models
- Selected a champion model using ROC-AUC
- Saved model metadata and monitoring artifacts
- Built a FastAPI scoring service
- Written tests
- Prepared a GitHub-ready project

This is not a notebook-only project. You are building a small machine learning service.

## Business Problem

Lenders need to estimate whether a loan applicant may default. A default means the borrower does not repay as expected.

Your model will predict:

```text
TARGET = 1 means default
TARGET = 0 means repaid
```

The output of the API will be a default probability and a risk band.

Example:

```json
{
  "default_probability": 0.12,
  "risk_band": "Low"
}
```

## Dataset

Dataset: Home Credit Default Risk  
Source: https://www.kaggle.com/competitions/home-credit-default-risk/data

You need these files:

| File | Purpose |
| --- | --- |
| `application_train.csv` | Main application table with the `TARGET` label |
| `bureau.csv` | External credit bureau history |
| `previous_application.csv` | Previous Home Credit applications |
| `HomeCredit_columns_description.csv` | Data dictionary |

The raw CSVs are large. Do not commit them to GitHub.

## Why This Project Matters

A weak credit risk project trains a model only on one table.

A stronger project:

- Uses real external data
- Validates data before training
- Handles class imbalance
- Joins historical credit behavior
- Uses ranking metrics, not just accuracy
- Exposes the model as an API
- Documents governance limits

That is what you will build here.

## Before You Start

Install:

- Python 3.11 or newer
- Git
- VS Code
- Kaggle account
- Kaggle API token, optional but helpful

Create your project:

```bash
mkdir credit-risk-api
cd credit-risk-api
```

Create folders:

```text
credit-risk-api/
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

## Module 1: Create the Environment

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

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Exercise

Run:

```bash
python -c "import pandas, sklearn, fastapi; print('credit risk environment ready')"
```

Expected output:

```text
credit risk environment ready
```

### Checkpoint Question

Why should `data/raw/` be ignored by Git?

## Module 2: Download and Place the Data

Download from Kaggle:

```bash
kaggle competitions download -c home-credit-default-risk
```

Unzip the files and place these inside `data/raw/`:

```text
data/raw/application_train.csv
data/raw/bureau.csv
data/raw/previous_application.csv
data/raw/HomeCredit_columns_description.csv
```

### Practice Task

Write down the file sizes. Which file is largest?

### Checkpoint Question

Why is `application_train.csv` the core table?

## Module 3: Understand the Label and Class Imbalance

Create `src/explore_data.py`:

```python
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

app = pd.read_csv(RAW / "application_train.csv")

print("Rows:", len(app))
print("Columns:", app.shape[1])
print("Default rate:")
print(app["TARGET"].value_counts(normalize=True))
```

Run:

```bash
python src/explore_data.py
```

Expected idea:

```text
Rows: about 307,511
Default rate: about 8%
```

### Simple Explanation

This is an imbalanced classification problem. Most applicants repay, and only a smaller group defaults.

Accuracy can be misleading. A model that predicts "repaid" for everyone would be about 92% accurate but useless for risk ranking.

Better metrics:

- ROC-AUC
- Average precision
- Brier score

### Exercise

Explain why accuracy is not enough for this project.

## Module 4: Write Data Validation

### Explanation

Before training, check that the raw data has the columns you expect.

Create `src/data_validation.py`:

```python
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MONITORING = ROOT / "monitoring"
MONITORING.mkdir(exist_ok=True)

REQUIRED_APP_COLUMNS = [
    "SK_ID_CURR",
    "TARGET",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
]


def validate():
    app = pd.read_csv(RAW / "application_train.csv")
    bureau = pd.read_csv(RAW / "bureau.csv")
    previous = pd.read_csv(RAW / "previous_application.csv")

    missing = [col for col in REQUIRED_APP_COLUMNS if col not in app.columns]
    report = {
        "application_rows": int(len(app)),
        "bureau_rows": int(len(bureau)),
        "previous_application_rows": int(len(previous)),
        "missing_required_application_columns": missing,
        "default_rate": round(float(app["TARGET"].mean()), 5),
        "application_missingness_top10": (
            app.isna().mean().sort_values(ascending=False).head(10).round(4).to_dict()
        ),
        "bureau_join_coverage": round(float(app["SK_ID_CURR"].isin(bureau["SK_ID_CURR"]).mean()), 4),
        "previous_join_coverage": round(float(app["SK_ID_CURR"].isin(previous["SK_ID_CURR"]).mean()), 4),
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

### Practice Task

Add one more validation check:

```text
Does TARGET contain only 0 and 1?
```

### Checkpoint Question

What could go wrong if you train without checking the raw files first?

## Module 5: Build Application-Level Features

### Explanation

Feature engineering turns raw columns into useful model inputs.

Examples:

- Age in years
- Employment years
- Credit-to-income ratio
- Annuity-to-income ratio
- Goods-to-credit ratio

Create `src/preprocess.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def safe_divide(a, b):
    return np.where((b == 0) | pd.isna(b), np.nan, a / b)


def application_features(app):
    df = app.copy()
    df["age_years"] = (-df["DAYS_BIRTH"]) / 365.25
    df["employment_years"] = np.where(
        df["DAYS_EMPLOYED"] == 365243,
        np.nan,
        (-df["DAYS_EMPLOYED"]) / 365.25,
    )
    df["credit_to_income_ratio"] = safe_divide(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["annuity_to_income_ratio"] = safe_divide(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["goods_to_credit_ratio"] = safe_divide(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
    return df
```

### Exercise

Add a new feature:

```text
children_per_family_member
```

Hint:

```python
df["children_per_family_member"] = safe_divide(df["CNT_CHILDREN"], df["CNT_FAM_MEMBERS"])
```

### Checkpoint Question

Why is `credit_to_income_ratio` more useful than `AMT_CREDIT` alone?

## Module 6: Aggregate Bureau History

### Explanation

The bureau table has multiple rows per applicant. The model needs one row per applicant.

So we aggregate:

- number of bureau records
- number of active accounts
- total credit
- total debt
- maximum overdue days

Add to `src/preprocess.py`:

```python
def bureau_features(bureau):
    b = bureau.copy()
    b["is_active"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)
    grouped = b.groupby("SK_ID_CURR").agg(
        bureau_count=("SK_ID_BUREAU", "count"),
        bureau_active_count=("is_active", "sum"),
        bureau_total_credit=("AMT_CREDIT_SUM", "sum"),
        bureau_total_debt=("AMT_CREDIT_SUM_DEBT", "sum"),
        bureau_total_overdue=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        bureau_max_days_overdue=("CREDIT_DAY_OVERDUE", "max"),
    ).reset_index()
    grouped["bureau_debt_to_credit_ratio"] = safe_divide(
        grouped["bureau_total_debt"],
        grouped["bureau_total_credit"],
    )
    return grouped
```

### Practice Task

Add:

```text
bureau_closed_count
```

Hint: create `is_closed` from `CREDIT_ACTIVE == "Closed"`.

### Checkpoint Question

Why must bureau records be aggregated before joining to `application_train.csv`?

## Module 7: Aggregate Previous Applications

### Explanation

Previous applications tell us whether the applicant had earlier approved or refused loans.

Add:

```python
def previous_application_features(previous):
    p = previous.copy()
    p["was_approved"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    p["was_refused"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    p["was_canceled"] = (p["NAME_CONTRACT_STATUS"] == "Canceled").astype(int)

    grouped = p.groupby("SK_ID_CURR").agg(
        prev_application_count=("SK_ID_PREV", "count"),
        prev_approved_count=("was_approved", "sum"),
        prev_refused_count=("was_refused", "sum"),
        prev_canceled_count=("was_canceled", "sum"),
        prev_total_credit=("AMT_CREDIT", "sum"),
        prev_avg_annuity=("AMT_ANNUITY", "mean"),
        prev_avg_down_payment=("AMT_DOWN_PAYMENT", "mean"),
    ).reset_index()

    grouped["prev_refusal_rate"] = safe_divide(
        grouped["prev_refused_count"],
        grouped["prev_application_count"],
    )
    return grouped
```

### Exercise

Why might `prev_refusal_rate` help predict risk?

## Module 8: Join the Tables

### Explanation

Now combine the three feature sets into one modeling table.

Add:

```python
def load_training_data():
    app = pd.read_csv(RAW / "application_train.csv")
    bureau = pd.read_csv(RAW / "bureau.csv")
    previous = pd.read_csv(RAW / "previous_application.csv")

    df = application_features(app)
    df = df.merge(bureau_features(bureau), on="SK_ID_CURR", how="left")
    df = df.merge(previous_application_features(previous), on="SK_ID_CURR", how="left")

    y = df["TARGET"]
    drop_cols = ["TARGET", "SK_ID_CURR"]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    return X, y
```

### Practice Task

Print:

```python
X, y = load_training_data()
print(X.shape)
print(y.mean())
```

Expected:

```text
About 307K rows
Default rate about 0.08
```

### Checkpoint Question

Why do we remove `TARGET` and `SK_ID_CURR` before training?

## Module 9: Build the Preprocessing Pipeline

### Explanation

Machine learning models need numeric arrays. But the dataset contains numeric and categorical columns.

We will use:

- median imputation for numeric columns
- most-frequent imputation for categorical columns
- one-hot encoding for categorical columns

Add:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessing_pipeline():
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=20)),
    ])

    def column_selector(X):
        numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_cols = [col for col in X.columns if col not in numeric_cols]
        return numeric_cols, categorical_cols

    class DynamicPreprocessor(ColumnTransformer):
        def fit(self, X, y=None):
            numeric_cols, categorical_cols = column_selector(X)
            self.transformers = [
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
            return super().fit(X, y)

    return DynamicPreprocessor([])
```

### Checkpoint Question

Why do we use `handle_unknown="ignore"` for one-hot encoding?

## Module 10: Train Candidate Models

### Explanation

Do not assume one model is best. Compare candidates.

Create `src/train.py`:

```python
import json
import os
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocess import build_preprocessing_pipeline, load_training_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MONITORING = ROOT / "monitoring"
MODEL_DIR.mkdir(exist_ok=True)
MONITORING.mkdir(exist_ok=True)


def build_candidates(scale_pos_weight):
    return {
        "logistic_regression": Pipeline([
            ("prep", build_preprocessing_pipeline()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]),
        "random_forest": Pipeline([
            ("prep", build_preprocessing_pipeline()),
            ("model", RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            )),
        ]),
        "xgboost": Pipeline([
            ("prep", build_preprocessing_pipeline()),
            ("model", XGBClassifier(
                n_estimators=250,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                eval_metric="auc",
                random_state=42,
            )),
        ]),
    }


def evaluate(model, X_test, y_test):
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": round(roc_auc_score(y_test, prob), 4),
        "avg_precision": round(average_precision_score(y_test, prob), 4),
        "brier_score": round(brier_score_loss(y_test, prob), 4),
    }


def train():
    X, y = load_training_data()

    sample_rows = int(os.getenv("TRAIN_SAMPLE_ROWS", "0"))
    if sample_rows and sample_rows < len(X):
        sample = X.assign(_target=y).sample(sample_rows, random_state=42)
        y = sample.pop("_target")
        X = sample

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    candidates = build_candidates(scale_pos_weight)
    results = {}

    for name, model in candidates.items():
        print(f"Training {name}")
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)

    champion_name = max(results, key=lambda name: results[name]["roc_auc"])
    champion = candidates[champion_name]

    joblib.dump(champion, MODEL_DIR / "credit_model.pkl")
    metadata = {
        "model_name": champion_name,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "default_rate": round(float(y.mean()), 5),
        "candidate_results": results,
        "features": list(X.columns),
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    train()
```

For a faster first run:

```bash
$env:TRAIN_SAMPLE_ROWS=20000
python src/train.py
```

### Exercise

Which model won by ROC-AUC?

### Checkpoint Question

Why do we use ROC-AUC instead of accuracy?

## Module 11: Add Risk Bands and Prediction Helpers

Create `src/predict.py`:

```python
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "credit_model.pkl"
META_PATH = ROOT / "models" / "model_metadata.json"


class ModelRegistry:
    model = None
    metadata = None

    @classmethod
    def load(cls):
        if cls.model is None:
            cls.model = joblib.load(MODEL_PATH)
        if cls.metadata is None:
            cls.metadata = json.loads(META_PATH.read_text())
        return cls.model, cls.metadata


def risk_band(probability):
    if probability >= 0.35:
        return "High"
    if probability >= 0.18:
        return "Medium"
    return "Low"


def predict(record):
    model, meta = ModelRegistry.load()
    frame = pd.DataFrame([record])
    probability = float(model.predict_proba(frame)[0, 1])
    return {
        "default_probability": round(probability, 4),
        "risk_band": risk_band(probability),
        "model_version": meta.get("model_name", "unknown"),
    }
```

### Practice Task

Change the risk bands to:

- High: 0.40 and above
- Medium: 0.20 to 0.40
- Low: below 0.20

Explain how this changes business decisions.

## Module 12: Build the FastAPI Service

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

from predict import ModelRegistry, predict

app = FastAPI(title="Home Credit Default Risk API")


class HomeCreditApplication(BaseModel):
    NAME_CONTRACT_TYPE: str = "Cash loans"
    CODE_GENDER: str = "F"
    FLAG_OWN_CAR: str = "N"
    FLAG_OWN_REALTY: str = "Y"
    CNT_CHILDREN: int = Field(0, ge=0)
    AMT_INCOME_TOTAL: float = Field(..., gt=0)
    AMT_CREDIT: float = Field(..., gt=0)
    AMT_ANNUITY: Optional[float] = Field(None, ge=0)
    AMT_GOODS_PRICE: Optional[float] = Field(None, ge=0)
    NAME_INCOME_TYPE: str = "Working"
    NAME_EDUCATION_TYPE: str = "Secondary / secondary special"
    NAME_FAMILY_STATUS: str = "Married"
    NAME_HOUSING_TYPE: str = "House / apartment"
    DAYS_BIRTH: int = Field(..., le=-1)
    DAYS_EMPLOYED: int
    EXT_SOURCE_1: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_2: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_3: Optional[float] = Field(None, ge=0, le=1)


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/readyz")
def ready():
    try:
        _, meta = ModelRegistry.load()
        return {"status": "ready", "model": meta["model_name"]}
    except Exception as exc:
        raise HTTPException(503, detail=str(exc))


@app.post("/v1/predict")
def score(application: HomeCreditApplication):
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

### Exercise

Add a batch prediction endpoint:

```text
POST /v1/predict/batch
```

It should accept a list of applications.

## Module 13: Write Tests

Create `tests/test_api.py`:

```python
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))


def test_health_endpoint():
    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)
    assert client.get("/healthz").json()["status"] == "ok"


def test_prediction_endpoint_with_mocked_model():
    from fastapi.testclient import TestClient
    import app as api_app

    sample_response = {
        "default_probability": 0.12,
        "risk_band": "Low",
        "model_version": "test-model",
    }

    with patch("app.predict", return_value=sample_response), patch(
        "app.ModelRegistry.load",
        return_value=(MagicMock(), {"model_name": "test-model"}),
    ):
        client = TestClient(api_app.app)
        response = client.post("/v1/predict", json={
            "AMT_INCOME_TOTAL": 162000,
            "AMT_CREDIT": 406597.5,
            "AMT_ANNUITY": 24700.5,
            "AMT_GOODS_PRICE": 351000,
            "DAYS_BIRTH": -12005,
            "DAYS_EMPLOYED": -4542,
            "EXT_SOURCE_2": 0.262949,
            "EXT_SOURCE_3": 0.139376
        })
        assert response.status_code == 200
        assert response.json()["risk_band"] == "Low"
```

Run:

```bash
pytest tests/ -q
```

### Practice Task

Add a test that rejects a negative `AMT_CREDIT`.

## Module 14: Add Monitoring Artifacts

### Explanation

A professional ML project should not only save a model. It should save evidence.

Useful artifacts:

- `model_metadata.json`
- ROC curve image
- Precision-recall curve image
- Score distribution plot
- Feature importance file
- Model card

Create `monitoring/model_card.md`:

```md
# Model Card: Home Credit Default Risk

## Intended Use

Estimate loan default probability for portfolio demonstration and credit risk workflow support.

## Data

Home Credit Default Risk Kaggle dataset.

## Metrics

Report ROC-AUC, average precision, and Brier score.

## Limitations

This model is not a production lending decision system. Real credit decisions require fairness review, compliance review, policy rules, and human governance.
```

### Exercise

Add your champion model name and validation metrics to the model card.

## Module 15: Write the README

Your README should include:

- Project title
- Business problem
- Dataset source
- Data setup instructions
- Feature engineering summary
- Model training instructions
- API instructions
- Metrics
- Governance notes
- Project structure

### Practice Task

Write a section called:

```text
Why Accuracy Is Not Enough
```

Explain class imbalance in your own words.

## Final Assignment

Extend the project with one new feature.

Choose one:

1. Add calibration curve monitoring
2. Add feature importance plot
3. Add SHAP explanations for top predictions
4. Add an endpoint that returns risk band counts for a batch
5. Add a fairness check comparing default scores by `CODE_GENDER`

Your final assignment must include:

- Code
- At least one test
- README update
- One monitoring artifact

## Final Rubric

| Area | Excellent |
| --- | --- |
| Data setup | Uses real Home Credit files and does not commit raw CSVs |
| Validation | Checks schema, class balance, missingness, and join coverage |
| Feature engineering | Uses application, bureau, and previous-application features |
| Modeling | Compares multiple models and selects champion by ROC-AUC |
| Metrics | Reports ROC-AUC, average precision, and Brier score |
| API | Provides health, readiness, and scoring endpoints |
| Tests | Tests API behavior and important validation rules |
| Governance | Explains limitations of credit scoring and human review |

## Interview Practice

Answer these aloud:

1. Why is this an imbalanced classification problem?
2. Why is ROC-AUC more useful than accuracy here?
3. What bureau features did you engineer?
4. Why must `SK_ID_CURR` be removed before training?
5. What does the Brier score measure?
6. How would you monitor this model in production?
7. What fairness risks exist in credit scoring?

## Finished Project Reference

The finished project version is available here:

https://github.com/iodsghana/credit-risk-api
