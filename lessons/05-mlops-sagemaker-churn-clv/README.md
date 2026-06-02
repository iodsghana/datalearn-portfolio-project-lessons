# Lesson 05: Build an MLOps Churn and CLV Platform

## What You Will Build

In this lesson, you will build a production-style MLOps project for customer churn and customer lifetime value.

By the end, you will have:

- Used the real IBM Telco Customer Churn dataset
- Engineered customer churn features
- Trained and compared churn models
- Tuned a recall-weighted threshold with F2 score
- Built a CLV model pattern
- Created customer segments
- Built a FastAPI scoring service
- Simulated a SageMaker training and model registry pipeline
- Added an evaluation gate
- Simulated champion/challenger deployment
- Built a PSI drift monitor
- Added Terraform-style infrastructure files
- Written tests

This lesson is about the complete lifecycle:

```text
train -> evaluate -> register -> deploy -> monitor -> retrain
```

## Dataset

Dataset: IBM Telco Customer Churn  
Common source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset has 7,043 customer records and a churn label.

Important columns:

| Column | Meaning |
| --- | --- |
| `customerID` | Customer identifier |
| `tenure` | Months with company |
| `MonthlyCharges` | Monthly bill |
| `TotalCharges` | Total billed amount |
| `Contract` | Month-to-month, one-year, or two-year |
| `PaymentMethod` | Billing method |
| `InternetService` | DSL, fiber optic, or no internet |
| `OnlineSecurity`, `TechSupport` | Service add-ons |
| `Churn` | Target label |

## Business Problem

Retention teams want to know:

1. Which customers are likely to churn?
2. Which customers are valuable enough to save?
3. Which retention offer has positive expected ROI?
4. How do we know if the deployed model is drifting?

The project has two modeling layers:

- Churn risk: probability that a customer leaves
- CLV: expected future customer value

## MLOps Problem

A model is not done when training finishes.

A production team also needs:

- Model registry
- Approval gates
- Deployment strategy
- Monitoring
- Drift alerts
- Retraining trigger
- Infrastructure definition
- Tests

That is what makes this an MLOps project.

## Before You Start

Create the project:

```bash
mkdir mlops-sagemaker-churn-clv
cd mlops-sagemaker-churn-clv
```

Create folders:

```text
mlops-sagemaker-churn-clv/
  api/
  data/
  infra/
    lambda/
  models/
  monitoring/
  pipelines/
  src/
  tests/
```

Create `.gitignore`:

```txt
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
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
boto3>=1.34,<2
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
python -c "import pandas, sklearn, boto3, fastapi; print('mlops environment ready')"
```

## Module 2: Understand the MLOps Architecture

The project architecture:

```text
IBM Telco data
  -> feature engineering
  -> churn + CLV training
  -> model artifacts and model card
  -> FastAPI scoring service
  -> SageMaker-compatible training/inference scripts
  -> model evaluation gate
  -> model registry
  -> champion/challenger deployment
  -> PSI drift monitor
  -> retraining trigger
```

### Checkpoint Question

Why is an evaluation gate important before registering or deploying a model?

## Module 3: Load and Validate Telco Data

Create `src/data_validation.py`:

```python
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MONITORING = ROOT / "monitoring"
MONITORING.mkdir(exist_ok=True)

REQUIRED_COLUMNS = [
    "customerID",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "PaymentMethod",
    "Churn",
]


def validate():
    df = pd.read_csv(DATA / "telco_churn.csv")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    churn_rate = (df["Churn"] == "Yes").mean()
    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_required_columns": missing,
        "churn_rate": round(float(churn_rate), 4),
        "missing_values_top10": df.isna().mean().sort_values(ascending=False).head(10).round(4).to_dict(),
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

Add a check that confirms `Churn` contains only `Yes` and `No`.

## Module 4: Engineer Churn Features

Create `src/churn_features.py`:

```python
import numpy as np
import pandas as pd

TARGET = "Churn"


class ChurnFeatureEngineer:
    PAYMENT_RISK = {
        "Electronic check": 3,
        "Mailed check": 2,
        "Bank transfer (automatic)": 1,
        "Credit card (automatic)": 1,
    }
    CONTRACT_RISK = {
        "Month-to-month": 3,
        "One year": 2,
        "Two year": 1,
    }
    INTERNET_TIER = {"No": 0, "DSL": 1, "Fiber optic": 2}

    def fit(self, X, y=None):
        total = pd.to_numeric(X["TotalCharges"], errors="coerce")
        self.total_charges_median_ = total.median()
        return self

    def transform(self, X):
        df = X.copy()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(self.total_charges_median_)

        df["tenure_years"] = df["tenure"] / 12
        df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
        df["is_mature"] = (df["tenure"] >= 36).astype(int)

        df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
        df["charge_ratio"] = df["MonthlyCharges"] / (df["avg_monthly_spend"] + 1e-6)
        df["spend_acceleration"] = df["MonthlyCharges"] - df["avg_monthly_spend"]

        service_cols = [
            "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        service_flags = []
        for col in service_cols:
            flag = f"has_{col.lower()}"
            df[flag] = (df[col] == "Yes").astype(int)
            service_flags.append(flag)

        df["service_count"] = df[service_flags].sum(axis=1)
        df["service_depth"] = df["service_count"] / len(service_cols)
        df["internet_tier"] = df["InternetService"].map(self.INTERNET_TIER).fillna(0)
        df["has_security"] = (df["OnlineSecurity"] == "Yes").astype(int)
        df["has_support"] = (df["TechSupport"] == "Yes").astype(int)

        df["contract_risk"] = df["Contract"].map(self.CONTRACT_RISK).fillna(2)
        df["payment_risk"] = df["PaymentMethod"].map(self.PAYMENT_RISK).fillna(2)
        df["combined_risk"] = df["contract_risk"] * df["payment_risk"]
        df["is_month2month"] = (df["Contract"] == "Month-to-month").astype(int)
        df["is_auto_pay"] = df["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int)

        df["engagement_score"] = (
            df["service_depth"] * 0.35
            + (1 - df["is_month2month"]) * 0.30
            + df["is_auto_pay"] * 0.20
            + df["is_mature"] * 0.15
        )
        df["tenure_x_monthly"] = df["tenure"] * df["MonthlyCharges"]
        df["risk_x_new"] = df["combined_risk"] * df["is_new_customer"]

        drop_cols = [
            "customerID", "gender", "Partner", "Dependents", "PhoneService",
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
            "Contract", "PaperlessBilling", "PaymentMethod",
        ] + service_flags
        return df.drop(columns=[c for c in drop_cols if c in df.columns]).fillna(0)
```

### Simple Explanation

Good churn features often describe:

- lifecycle: new or mature customer
- billing: high charges, price sensitivity
- service depth: how many services the customer uses
- contract risk: month-to-month customers can leave more easily
- support coverage: lack of tech support/security can increase churn

### Practice Task

Add a feature:

```text
high_monthly_charge = MonthlyCharges > 80
```

## Module 5: Train Champion-Challenger Churn Models

Create `src/train_churn_clv.py`:

```python
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from churn_features import ChurnFeatureEngineer, TARGET

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco_churn.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def optimise_threshold(y_true, y_prob, beta=2.0):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)
    best = int(np.argmax(f[:-1]))
    return float(thresholds[best]), float(f[best])


def train_churn_model():
    df = pd.read_csv(DATA_PATH)
    y = (df[TARGET] == "Yes").astype(int)
    engineer = ChurnFeatureEngineer()
    X = engineer.fit(df).transform(df.drop(columns=[TARGET, "customerID"], errors="ignore"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    candidates = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=250, max_depth=8, class_weight="balanced", random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "xgboost": XGBClassifier(eval_metric="aucpr", random_state=42, verbosity=0),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        cv_scores[name] = float(scores.mean())

    champion_name = max(cv_scores, key=cv_scores.get)
    champion = candidates[champion_name]
    champion.fit(X_train, y_train)
    prob = champion.predict_proba(X_test)[:, 1]
    threshold, f2 = optimise_threshold(y_test, prob)

    metrics = {
        "champion_name": champion_name,
        "cv_scores": {k: round(v, 4) for k, v in cv_scores.items()},
        "test_auc": round(roc_auc_score(y_test, prob), 4),
        "test_avg_precision": round(average_precision_score(y_test, prob), 4),
        "threshold": round(threshold, 4),
        "f2_at_threshold": round(f2, 4),
        "feature_count": int(X.shape[1]),
    }
    artifact = {
        "churn_model": champion,
        "feature_engineer": engineer,
        "feature_names": list(X.columns),
        "threshold": threshold,
    }
    joblib.dump(artifact, MODEL_DIR / "churn_clv_model.pkl")
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train_churn_model()
```

Run:

```bash
python src/train_churn_clv.py
```

### Exercise

Which model became champion? What was its ROC-AUC?

### Checkpoint Question

Why might a churn team optimize F2 instead of plain accuracy?

## Module 6: Add CLV Logic

### Explanation

CLV means customer lifetime value. A simple first version can estimate future value from monthly charges and churn risk.

Add to `src/churn_predict.py` later:

```python
def estimate_clv(monthly_charges, churn_probability, months=12):
    retention_probability = 1 - churn_probability
    return round(monthly_charges * months * retention_probability, 2)
```

### Practice Task

If a customer pays `$80/month` and has churn probability `0.25`, what is simple 12-month CLV?

Answer:

```text
80 * 12 * 0.75 = $720
```

### Advanced Note

The finished project uses BG/NBD and Gamma-Gamma CLV models. Those are stronger probabilistic CLV methods, but this simple CLV formula helps you understand the business logic first.

## Module 7: Add Segmentation

### Explanation

Segments help a business choose actions.

Example rules:

| Segment | Meaning |
| --- | --- |
| `Champions` | High engagement, low churn risk |
| `At Risk` | High churn risk |
| `Loyal` | Long-tenure, stable customer |
| `New Customers` | New relationship |

Add:

```python
def segment_customer(churn_probability, tenure, engagement_score):
    if churn_probability >= 0.55:
        return "At Risk"
    if engagement_score >= 0.65 and tenure >= 30:
        return "Champions"
    if tenure >= 24 and churn_probability < 0.25:
        return "Loyal"
    if tenure < 10:
        return "New Customers"
    return "Needs Attention"
```

### Exercise

Create one extra segment:

```text
High Value Save
```

Use churn probability and CLV together.

## Module 8: Build Prediction Helpers

Create `src/churn_predict.py`:

```python
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_clv_model.pkl"
META_PATH = ROOT / "models" / "model_metadata.json"


class ChurnCLVRegistry:
    artifact = None
    metadata = None

    @classmethod
    def load(cls):
        if cls.artifact is None:
            cls.artifact = joblib.load(MODEL_PATH)
        if cls.metadata is None:
            cls.metadata = json.loads(META_PATH.read_text())
        return cls.artifact, cls.metadata


def risk_tier(probability):
    if probability >= 0.60:
        return "Very High"
    if probability >= 0.35:
        return "High"
    if probability >= 0.15:
        return "Medium"
    return "Low"


def recommended_action(probability):
    if probability >= 0.60:
        return "Call customer and offer premium retention package"
    if probability >= 0.35:
        return "Send targeted discount or service support offer"
    if probability >= 0.15:
        return "Monitor and send engagement campaign"
    return "No immediate action"


def predict_churn(record):
    artifact, meta = ChurnCLVRegistry.load()
    frame = pd.DataFrame([record])
    X = artifact["feature_engineer"].transform(frame)
    probability = float(artifact["churn_model"].predict_proba(X)[0, 1])
    return {
        "churn_probability": round(probability, 4),
        "risk_tier": risk_tier(probability),
        "recommended_action": recommended_action(probability),
        "model_version": meta.get("sha256", "local")[:12],
    }


def predict_clv(record):
    churn = predict_churn(record)
    monthly = float(record.get("MonthlyCharges", 0))
    probability = churn["churn_probability"]
    clv_12m = round(monthly * 12 * (1 - probability), 2)
    clv_24m = round(monthly * 24 * (1 - probability), 2)
    return {"clv_12m": clv_12m, "clv_24m": clv_24m, "p_alive": round(1 - probability, 4)}
```

### Practice Task

Add a `score_full(record)` function that returns churn and CLV together.

## Module 9: Add Retention ROI

### Explanation

A retention offer should be worth more than it costs.

Add:

```python
def retention_roi(record, offer_discount_pct=0.10):
    churn = predict_churn(record)
    clv = predict_clv(record)
    monthly = float(record.get("MonthlyCharges", 0))
    offer_cost = monthly * 12 * offer_discount_pct
    expected_saved_value = churn["churn_probability"] * clv["clv_12m"]
    net_benefit = expected_saved_value - offer_cost
    return {
        "expected_saved_value": round(expected_saved_value, 2),
        "offer_cost": round(offer_cost, 2),
        "net_benefit": round(net_benefit, 2),
        "offer_recommended": net_benefit > 0,
    }
```

### Exercise

Why might a high-churn, low-CLV customer not receive an expensive retention offer?

## Module 10: Build the FastAPI Service

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

from churn_predict import ChurnCLVRegistry, predict_churn, predict_clv, retention_roi

app = FastAPI(title="Customer Churn + CLV Engine API")


class CustomerRecord(BaseModel):
    tenure: int = Field(..., ge=0, le=120)
    MonthlyCharges: float = Field(..., ge=0, le=500)
    TotalCharges: Optional[float] = None
    SeniorCitizen: int = Field(0, ge=0, le=1)
    gender: str = "Male"
    Partner: str = "No"
    Dependents: str = "No"
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"


class RetentionROIRequest(BaseModel):
    customer: CustomerRecord
    offer_discount_pct: float = Field(0.10, ge=0.01, le=0.50)


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/readyz")
def ready():
    try:
        _, meta = ChurnCLVRegistry.load()
        return {"status": "ready", "churn_auc": meta.get("test_auc")}
    except Exception as exc:
        raise HTTPException(503, detail=str(exc))


@app.post("/v1/churn/predict")
def churn(customer: CustomerRecord):
    start = time.perf_counter()
    result = predict_churn(customer.model_dump())
    return {"request_id": str(uuid.uuid4()), "latency_ms": round((time.perf_counter() - start) * 1000, 2), **result}


@app.post("/v1/clv/predict")
def clv(customer: CustomerRecord):
    return predict_clv(customer.model_dump())


@app.post("/v1/retention/roi")
def roi(req: RetentionROIRequest):
    return retention_roi(req.customer.model_dump(), req.offer_discount_pct)
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
POST /v1/score
```

It should return churn probability, CLV, and segment.

## Module 11: Create a SageMaker-Compatible Training Script

### Explanation

SageMaker training containers need a script that can read training data, train a model, and write artifacts.

Create `src/train.py`:

```python
import argparse
from pathlib import Path

from train_churn_clv import train_churn_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/opt/ml/model")
    parser.add_argument("--train", default="/opt/ml/input/data/train")
    args = parser.parse_args()

    train_churn_model()
    print(f"Training complete. Model artifacts should be copied to {args.model_dir}")


if __name__ == "__main__":
    main()
```

### Checkpoint Question

Why do cloud training jobs need predictable input and output paths?

## Module 12: Create a SageMaker Inference Script

Create `src/inference.py`:

```python
import json

from churn_predict import predict_churn


def model_fn(model_dir):
    return None


def input_fn(request_body, content_type):
    if content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    return predict_churn(input_data)


def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
```

### Practice Task

Add support for CSV input.

## Module 13: Simulate the SageMaker Pipeline

Create `pipelines/sagemaker_pipeline.py`:

```python
import os
from datetime import datetime

SIMULATE = os.environ.get("SIMULATE_AWS", "true").lower() == "true"

PIPELINE_CONFIG = {
    "project": "mlops-churn-clv",
    "auc_gate": 0.80,
    "champion_traffic": 90,
    "psi_alert_threshold": 0.10,
}


def validate_data():
    return {
        "schema_ok": True,
        "null_rate_max": 0.03,
        "class_balance": 0.265,
        "psi_vs_baseline": 0.062,
        "validation_passed": True,
    }


def launch_training_job():
    job_name = f"{PIPELINE_CONFIG['project']}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    metrics = {"val_auc": 0.8405, "val_ap": 0.6510, "psi_baseline": 0.048}
    return job_name, "s3://demo/models/model.tar.gz", metrics


def evaluate_model(metrics):
    auc = metrics["val_auc"]
    if auc < PIPELINE_CONFIG["auc_gate"]:
        raise RuntimeError(f"AUC {auc} below gate")
    return True


def register_model(job_name, model_s3, metrics):
    return {
        "model_package_arn": f"arn:aws:sagemaker:demo:model-package/{job_name}",
        "approval_status": "PendingManualApproval",
        "metrics": metrics,
    }


def deploy_ab_endpoint(model_name):
    return {
        "endpoint": "churn-clv-endpoint",
        "champion_weight": PIPELINE_CONFIG["champion_traffic"],
        "challenger_weight": 100 - PIPELINE_CONFIG["champion_traffic"],
    }


def run_pipeline():
    validation = validate_data()
    job_name, model_s3, metrics = launch_training_job()
    evaluate_model(metrics)
    registry = register_model(job_name, model_s3, metrics)
    endpoint = deploy_ab_endpoint(job_name)
    return {"validation": validation, "registry": registry, "endpoint": endpoint}


if __name__ == "__main__":
    print(run_pipeline())
```

Run:

```bash
$env:SIMULATE_AWS="true"
python pipelines/sagemaker_pipeline.py
```

### Exercise

Change `auc_gate` to `0.90`. What happens?

## Module 14: Add PSI Drift Monitoring

Create `monitoring/drift_monitor.py`:

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd

PSI_WARNING = 0.10
PSI_CRITICAL = 0.25


def psi_score(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] -= 1e-6
    breakpoints[-1] += 1e-6
    exp_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual, bins=breakpoints)[0] / max(len(actual), 1)
    exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-6, act_pct)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_status(score):
    if score < PSI_WARNING:
        return "stable"
    if score < PSI_CRITICAL:
        return "warning"
    return "critical"


def run_drift_check():
    rng = np.random.default_rng(42)
    baseline = rng.beta(0.5, 5, 1000)
    current = rng.beta(0.7, 4, 1000)
    score = psi_score(baseline, current)
    report = {
        "score_psi": round(score, 4),
        "status": psi_status(score),
        "retraining_triggered": score >= PSI_CRITICAL,
    }
    Path("monitoring").mkdir(exist_ok=True)
    Path("monitoring/drift_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(run_drift_check())
```

### Simple Explanation

PSI compares two distributions:

- baseline: what the model saw during training
- current: what the model is seeing now

If they become too different, the model may be stale.

### Practice Task

Simulate severe drift and confirm the status becomes `critical`.

## Module 15: Add Terraform Infrastructure

Create `infra/main.tf`:

```hcl
terraform {
  required_version = ">= 1.5.0"
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

resource "aws_s3_bucket" "mlops_bucket" {
  bucket = "mlops-churn-clv-demo-bucket"
}

resource "aws_ecr_repository" "model_repo" {
  name = "mlops-churn-clv"
}

resource "aws_sns_topic" "retrain_topic" {
  name = "mlops-churn-clv-retrain"
}
```

### Checkpoint Question

Why is infrastructure-as-code useful for MLOps?

## Module 16: Write Tests

Create `tests/test_pipeline.py`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipelines"))
sys.path.insert(0, str(ROOT / "monitoring"))

from sagemaker_pipeline import evaluate_model, run_pipeline
from drift_monitor import psi_score, psi_status


def test_model_gate_passes():
    assert evaluate_model({"val_auc": 0.85}) is True


def test_model_gate_fails():
    try:
        evaluate_model({"val_auc": 0.70})
    except RuntimeError:
        assert True
    else:
        raise AssertionError("Expected model gate failure")


def test_pipeline_simulation_runs():
    result = run_pipeline()
    assert "registry" in result
    assert result["endpoint"]["champion_weight"] == 90


def test_psi_status():
    assert psi_status(0.05) == "stable"
    assert psi_status(0.15) == "warning"
    assert psi_status(0.30) == "critical"
```

Run:

```bash
pytest tests/ -q
```

### Practice Task

Add an API test for `/healthz`.

## Module 17: Write the Model Card

Create `monitoring/model_card.md`:

```md
# Model Card: Churn + CLV Platform

## Intended Use

Predict customer churn, estimate customer lifetime value, and support retention campaign prioritization.

## Dataset

IBM Telco Customer Churn dataset.

## Models

- Churn classifier
- CLV estimator
- Customer segmentation

## Evaluation

Report ROC-AUC, average precision, F2 threshold behavior, and CLV summary statistics.

## MLOps Controls

- Evaluation gate before registration
- Pending manual approval in registry
- Champion/challenger deployment
- PSI drift monitoring
- Retraining trigger

## Governance Notes

Retention recommendations should be reviewed for fairness, customer experience, and business policy.
```

### Exercise

Add your actual model metrics to the model card.

## Module 18: Write the README

Your README should include:

- Project title
- Dataset source
- Business problem
- Model architecture
- MLOps lifecycle
- API endpoints
- Local quickstart
- Pipeline simulation command
- Drift monitoring explanation
- Terraform notes
- Tests

### Practice Task

Write a section called:

```text
Why This Is MLOps, Not Just ML
```

## Final Assignment

Choose one extension:

1. Add a batch scoring endpoint.
2. Add a model approval JSON file that blocks deployment if status is not `Approved`.
3. Add a drift report endpoint to the API.
4. Add a Lambda handler that calls `run_pipeline()`.
5. Add a challenger model and compare champion/challenger metrics.

Your final assignment must include:

- Code
- One test
- README update
- MLOps explanation

## Final Rubric

| Area | Excellent |
| --- | --- |
| Data | Uses real IBM Telco churn data |
| Features | Builds lifecycle, billing, contract, service, and engagement features |
| Modeling | Compares models and selects a champion |
| Thresholding | Uses F2 or another justified retention threshold |
| CLV | Estimates future customer value and connects it to retention ROI |
| API | Serves churn, CLV, ROI, and model metadata |
| MLOps pipeline | Includes validation, training, gate, registry, deployment, monitoring |
| Drift | Computes PSI and defines retraining action |
| Infrastructure | Includes Terraform-style AWS resources |
| Tests | Covers pipeline gates, drift, features, and API |

## Interview Practice

Answer these aloud:

1. What is the difference between ML and MLOps?
2. Why do we need a model registry?
3. What is a champion/challenger deployment?
4. Why is F2 useful for churn?
5. How does CLV change retention decisions?
6. What is PSI?
7. When should drift trigger retraining?
8. Why should model approval require human review?
9. How would you deploy this on AWS SageMaker?

## Finished Project Reference

The finished project version is available here:

https://github.com/iodsghana/MLOps-SageMaker
