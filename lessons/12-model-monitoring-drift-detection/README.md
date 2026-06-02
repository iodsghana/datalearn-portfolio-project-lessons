# Lesson 12 - Model Monitoring and Drift Detection

In this lesson, you will learn how to monitor a machine learning model after it has been trained and served.

Training a model is not the end of the work. In real systems, data changes, user behavior changes, business rules change, and model performance can degrade.

This lesson teaches you how to detect those changes and explain them professionally.

You can apply this lesson to:

- Credit risk models
- Fraud detection models
- Churn models
- Student loan forecasting models
- NLP sentiment models
- Taxi demand forecasting models

## Learning Goals

By the end of this lesson, you should be able to:

- Explain what model drift is
- Separate data drift, prediction drift, and performance drift
- Create a reference dataset
- Compare a new batch against the reference batch
- Calculate Population Stability Index
- Monitor prediction distribution
- Track missing-value rates
- Create alert rules
- Write a monitoring report
- Explain model monitoring in an interview

## Key Concepts

There are three common monitoring categories.

| Category | Meaning | Example |
| --- | --- | --- |
| Data drift | Input data changes | Applicant income distribution changes |
| Prediction drift | Model outputs change | More borrowers are scored high risk |
| Performance drift | Accuracy or error gets worse | Default predictions become less reliable |

Performance drift requires new true labels. Data drift and prediction drift can be checked before labels arrive.

## Module 1 - Choose a Project

Pick one model project.

Recommended choices:

- Credit Risk API
- Fraud Detection API
- MLOps SageMaker Churn and CLV
- NLP Earnings Sentiment API

Create this folder:

```text
monitoring/
  reference_profile.json
  drift_report.md
  drift_report.json
```

Create this script:

```text
src/monitor_drift.py
```

### Practice

Answer:

1. What is the model predicting?
2. What features are most important?
3. Which inputs might change over time?
4. When would true labels become available?

## Module 2 - Create a Reference Dataset

A reference dataset is the baseline you compare future data against.

For example:

- Training data
- Validation data
- First stable production month
- A clean historical sample

Create a profile from reference data:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def numeric_profile(df: pd.DataFrame, columns: list[str]) -> dict:
    profile = {}

    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")
        profile[col] = {
            "count": int(series.notna().sum()),
            "missing_rate": round(float(series.isna().mean()), 4),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "p25": round(float(series.quantile(0.25)), 4),
            "p50": round(float(series.quantile(0.50)), 4),
            "p75": round(float(series.quantile(0.75)), 4),
            "max": round(float(series.max()), 4),
        }

    return profile
```

### Practice

Choose five numeric columns from your project.

Examples for credit risk:

- `AMT_CREDIT`
- `AMT_INCOME_TOTAL`
- `DAYS_BIRTH`
- `EXT_SOURCE_2`
- `bureau_credit_sum`

Examples for NLP:

- Text length
- Word count
- Positive probability
- Negative probability
- Confidence

## Module 3 - Save the Reference Profile

Add:

```python
def save_reference_profile(data_path: str, columns: list[str]) -> Path:
    df = pd.read_csv(data_path)
    profile = {
        "data_path": data_path,
        "row_count": int(len(df)),
        "columns": columns,
        "numeric_profile": numeric_profile(df, columns),
    }

    out = ROOT / "monitoring" / "reference_profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return out
```

Example:

```python
if __name__ == "__main__":
    save_reference_profile(
        data_path="data/processed/training_features.csv",
        columns=["AMT_CREDIT", "AMT_INCOME_TOTAL", "EXT_SOURCE_2"],
    )
```

### Exercise

Run your script and confirm:

```text
monitoring/reference_profile.json
```

exists.

## Module 4 - Calculate Population Stability Index

Population Stability Index, or PSI, compares two distributions.

Interpretation:

| PSI | Meaning |
| ---: | --- |
| < 0.10 | Low drift |
| 0.10 to 0.25 | Moderate drift |
| > 0.25 | High drift |

Add:

```python
import numpy as np


def psi(expected, actual, bins: int = 10) -> float:
    expected = pd.to_numeric(pd.Series(expected), errors="coerce").dropna()
    actual = pd.to_numeric(pd.Series(actual), errors="coerce").dropna()

    if expected.empty or actual.empty:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(expected.quantile(quantiles).to_numpy())

    if len(edges) < 3:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)

    expected_pct = np.maximum(expected_counts / max(expected_counts.sum(), 1), 0.0001)
    actual_pct = np.maximum(actual_counts / max(actual_counts.sum(), 1), 0.0001)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
```

### Practice

Create two simple arrays:

```python
baseline = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
new_batch = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
```

Calculate PSI.

Then create:

```python
new_batch = [10, 11, 12, 13, 15, 16, 17, 18]
```

Compare the result.

## Module 5 - Compare Reference and Current Data

Add:

```python
def compare_batches(reference_df: pd.DataFrame, current_df: pd.DataFrame, columns: list[str]) -> dict:
    results = {}

    for col in columns:
        ref = pd.to_numeric(reference_df[col], errors="coerce")
        cur = pd.to_numeric(current_df[col], errors="coerce")

        results[col] = {
            "reference_missing_rate": round(float(ref.isna().mean()), 4),
            "current_missing_rate": round(float(cur.isna().mean()), 4),
            "reference_mean": round(float(ref.mean()), 4),
            "current_mean": round(float(cur.mean()), 4),
            "mean_change": round(float(cur.mean() - ref.mean()), 4),
            "psi": round(psi(ref, cur), 4),
        }

    return results
```

### Exercise

Split your data into two batches:

```python
df = pd.read_csv("data/processed/training_features.csv")
reference_df = df.sample(frac=0.5, random_state=42)
current_df = df.drop(reference_df.index)
```

Then compare the batches.

## Module 6 - Add Alert Rules

Monitoring becomes useful when it creates action.

Add:

```python
def assign_alerts(drift_results: dict) -> list[dict]:
    alerts = []

    for feature, stats in drift_results.items():
        if stats["psi"] > 0.25:
            alerts.append({
                "feature": feature,
                "severity": "high",
                "reason": "PSI above 0.25",
                "psi": stats["psi"],
            })
        elif stats["psi"] > 0.10:
            alerts.append({
                "feature": feature,
                "severity": "medium",
                "reason": "PSI above 0.10",
                "psi": stats["psi"],
            })

        missing_change = stats["current_missing_rate"] - stats["reference_missing_rate"]
        if missing_change > 0.10:
            alerts.append({
                "feature": feature,
                "severity": "medium",
                "reason": "Missing rate increased by more than 10 percentage points",
                "missing_rate_change": round(missing_change, 4),
            })

    return alerts
```

### Practice

Answer:

1. Why should high PSI not automatically retrain a model?
2. What should a human review before retraining?
3. What alert would matter most in your project?

## Module 7 - Monitor Predictions

For classification models, track:

- Average predicted probability
- Percent high risk
- Class distribution
- Average confidence
- Low-confidence rate

Example:

```python
def prediction_profile(predictions: pd.Series, threshold: float = 0.5) -> dict:
    scores = pd.to_numeric(predictions, errors="coerce").dropna()

    return {
        "count": int(len(scores)),
        "mean_score": round(float(scores.mean()), 4),
        "p50_score": round(float(scores.quantile(0.5)), 4),
        "p90_score": round(float(scores.quantile(0.9)), 4),
        "high_risk_rate": round(float((scores >= threshold).mean()), 4),
    }
```

For NLP sentiment models, track:

```python
def class_distribution(labels: pd.Series) -> dict:
    counts = labels.value_counts(normalize=True)
    return {str(label): round(float(value), 4) for label, value in counts.items()}
```

### Exercise

Create a prediction profile for one batch of model outputs.

Write one sentence:

```text
The current batch has a high-risk rate of X, compared with Y in the reference period.
```

## Module 8 - Create a Drift Report

Create:

```python
def write_drift_report(results: dict, alerts: list[dict]) -> Path:
    lines = [
        "# Drift Report",
        "",
        "## Summary",
        "",
        f"- Features checked: {len(results)}",
        f"- Alerts generated: {len(alerts)}",
        "",
        "## Feature Drift",
        "",
        "| Feature | PSI | Reference Missing | Current Missing | Mean Change |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for feature, stats in results.items():
        lines.append(
            f"| {feature} | {stats['psi']} | {stats['reference_missing_rate']} | "
            f"{stats['current_missing_rate']} | {stats['mean_change']} |"
        )

    lines.extend(["", "## Alerts", ""])

    if alerts:
        for alert in alerts:
            lines.append(f"- {alert['severity'].upper()}: {alert['feature']} - {alert['reason']}")
    else:
        lines.append("- No alerts generated.")

    out = ROOT / "monitoring" / "drift_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
```

Also save JSON:

```python
def save_drift_json(results: dict, alerts: list[dict]) -> Path:
    out = ROOT / "monitoring" / "drift_report.json"
    out.write_text(
        json.dumps({"results": results, "alerts": alerts}, indent=2),
        encoding="utf-8",
    )
    return out
```

### Practice

Run the script and open:

```text
monitoring/drift_report.md
```

Explain the report to a classmate or friend.

## Module 9 - Add Drift Tests

Create tests:

```python
from src.monitor_drift import psi, assign_alerts


def test_psi_detects_large_shift():
    reference = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    current = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

    assert psi(reference, current) > 0.1


def test_alerts_created_for_high_psi():
    results = {
        "feature_a": {
            "psi": 0.3,
            "reference_missing_rate": 0.0,
            "current_missing_rate": 0.0,
        }
    }

    alerts = assign_alerts(results)

    assert len(alerts) == 1
    assert alerts[0]["severity"] == "high"
```

### Exercise

Add one test for missing-rate drift.

## Module 10 - Add Monitoring to the README

Add:

```markdown
## Monitoring

This project includes a drift monitoring workflow that compares a reference batch with a current batch.

The monitoring script checks:

- Numeric feature drift using PSI
- Missing-value rate changes
- Prediction distribution changes
- Alert thresholds for medium and high drift

Run:

```bash
python src/monitor_drift.py
```

Outputs:

- `monitoring/reference_profile.json`
- `monitoring/drift_report.json`
- `monitoring/drift_report.md`
```

### Practice

Add one screenshot or excerpt from the drift report to the README.

## Module 11 - Decide What Happens After an Alert

An alert is not the same as a fix.

Possible actions:

- Review data pipeline for errors
- Check if a source system changed
- Compare business seasonality
- Wait for true labels
- Re-evaluate performance
- Retrain model
- Roll back model
- Update feature logic

Create:

```text
monitoring/alert_response_playbook.md
```

Add:

```markdown
# Alert Response Playbook

## Medium Drift

1. Review drift report.
2. Check whether the feature changed due to expected business seasonality.
3. Inspect missing-value rates.
4. Monitor the next batch before retraining.

## High Drift

1. Review pipeline logs.
2. Check source data changes.
3. Compare prediction distribution.
4. Run model evaluation if labels are available.
5. Decide whether retraining or rollback is needed.

## Do Not Automatically Retrain When

- Labels are unavailable
- The shift is caused by a data bug
- The drift is expected seasonality
- The new data quality is poor
```

### Exercise

Add project-specific response steps.

## Final Assignment

Add drift monitoring to one project.

Required:

- `src/monitor_drift.py`
- `monitoring/reference_profile.json`
- `monitoring/drift_report.md`
- `monitoring/drift_report.json`
- PSI calculation
- Missing-rate drift checks
- Alert rules
- At least two drift tests
- README monitoring section
- Alert response playbook

Optional:

- Prediction distribution monitoring
- Charts
- Scheduled monitoring notes
- GitHub Actions workflow for monitoring checks
- Model retraining trigger design

## Rubric

| Area | Strong Submission |
| --- | --- |
| Reference Data | Clear baseline dataset or profile |
| Drift Metrics | PSI and missing-rate changes are calculated correctly |
| Alerts | Alert rules are clear and not too sensitive |
| Reports | Markdown and JSON reports are generated |
| Tests | Drift logic is tested |
| Interpretation | README explains what alerts mean |
| Governance | Alert response playbook avoids blind retraining |

## Interview Questions

Practice answering:

1. What is data drift?
2. What is prediction drift?
3. What is performance drift?
4. Why can data drift happen before performance labels arrive?
5. What is PSI?
6. When would you retrain a model?
7. Why should retraining not be automatic?
8. What would you monitor for a credit risk model?
9. What would you monitor for an NLP model?
10. How would you explain a drift alert to a business stakeholder?

## Completion Checklist

You are done when:

- Reference profile exists
- Drift script runs
- Drift report is generated
- PSI is calculated
- Missing-rate changes are checked
- Alerts are generated when thresholds are crossed
- Tests pass
- README explains monitoring
- Alert response playbook exists
- You can explain drift in plain English

