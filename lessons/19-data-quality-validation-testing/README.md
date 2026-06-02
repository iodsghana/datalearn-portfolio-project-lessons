# Lesson 19 - Data Quality, Validation, and Testing

In this lesson, you will learn how to check whether your data is trustworthy before using it for analytics, modeling, dashboards, or APIs.

Bad data can make a polished project misleading. A professional data project should include validation checks that catch missing columns, impossible values, duplicate records, leakage risks, and broken pipeline outputs.

## Learning Goals

By the end of this lesson, you should be able to:

- Define required schema rules
- Check missing values and duplicate records
- Validate numeric ranges and categorical values
- Detect possible target leakage
- Create a data quality report
- Add validation tests to a project
- Explain data quality checks in an interview

## What You Are Building

You will add a validation package to one existing project.

Final artifacts:

```text
src/validate_data.py
monitoring/data_quality_report.md
monitoring/data_quality_report.json
tests/test_data_quality.py
```

## Module 1 - Choose a Dataset

Pick one project dataset:

- Home Credit application data
- Fraud detection dataset
- Student loan forecasting data
- NYC taxi trip data
- Financial PhraseBank text data
- Online Retail data

Create:

```text
src/validate_data.py
monitoring/
tests/
```

### Practice

Answer:

1. What file is the main dataset?
2. What columns must exist?
3. Which column is the target or main output?
4. Which values would be impossible or suspicious?

## Module 2 - Define Schema Rules

Create `src/validate_data.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


REQUIRED_COLUMNS = [
    "SK_ID_CURR",
    "TARGET",
]


def check_required_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return sorted(set(required) - set(df.columns))
```

For your project, replace `REQUIRED_COLUMNS`.

Examples:

Financial NLP:

```python
REQUIRED_COLUMNS = ["text", "label", "label_name"]
```

NYC Taxi:

```python
REQUIRED_COLUMNS = ["pickup_datetime", "dropoff_datetime", "PULocationID", "DOLocationID"]
```

### Exercise

Write the required schema for one dataset.

Include:

- ID column
- Date column if present
- Target column if present
- Main feature columns

## Module 3 - Check Missing Values

Add:

```python
def missing_profile(df: pd.DataFrame) -> dict:
    rates = df.isna().mean().sort_values(ascending=False)
    return {
        column: round(float(rate), 4)
        for column, rate in rates.items()
        if rate > 0
    }
```

### Practice

Run this on your dataset.

Answer:

1. Which column has the highest missing rate?
2. Is that column important?
3. Should it be filled, dropped, or monitored?

## Module 4 - Check Duplicates

Add:

```python
def duplicate_count(df: pd.DataFrame, key_columns: list[str] | None = None) -> int:
    if key_columns:
        return int(df.duplicated(subset=key_columns).sum())
    return int(df.duplicated().sum())
```

Examples:

```python
duplicate_count(df)
duplicate_count(df, ["SK_ID_CURR"])
```

### Exercise

Check duplicates:

- Full-row duplicates
- Duplicate primary keys

Write whether duplicates are expected or suspicious.

## Module 5 - Validate Numeric Ranges

Add:

```python
def range_violations(df: pd.DataFrame, rules: dict[str, tuple[float | None, float | None]]) -> dict:
    violations = {}

    for column, (min_value, max_value) in rules.items():
        if column not in df.columns:
            continue

        series = pd.to_numeric(df[column], errors="coerce")
        bad = pd.Series(False, index=df.index)

        if min_value is not None:
            bad = bad | (series < min_value)
        if max_value is not None:
            bad = bad | (series > max_value)

        violations[column] = int(bad.sum())

    return violations
```

Example rules:

```python
RANGE_RULES = {
    "TARGET": (0, 1),
    "AMT_INCOME_TOTAL": (0, None),
    "AMT_CREDIT": (0, None),
}
```

### Practice

Create at least three numeric range rules.

Examples:

- Loan amount must be positive
- Probability must be between 0 and 1
- Trip distance must be nonnegative
- Text length must be greater than 0

## Module 6 - Validate Categorical Values

Add:

```python
def category_violations(df: pd.DataFrame, rules: dict[str, set]) -> dict:
    violations = {}

    for column, allowed_values in rules.items():
        if column not in df.columns:
            continue

        observed = set(df[column].dropna().unique())
        violations[column] = sorted(observed - allowed_values)

    return violations
```

Example:

```python
CATEGORY_RULES = {
    "label_name": {"negative", "neutral", "positive"},
}
```

### Exercise

Create one categorical rule for your dataset.

If your project has no categorical columns, create a rule for a label column or status column.

## Module 7 - Check Target Leakage

Target leakage happens when a feature contains information that would not be available at prediction time.

Examples:

- Using repayment outcome to predict default
- Using future trip counts to forecast current demand
- Using post-decision fraud labels as model features
- Using future sentiment market reaction for current text classification

Create a simple leakage scan:

```python
LEAKAGE_KEYWORDS = [
    "target",
    "default",
    "outcome",
    "label",
    "future",
    "post",
]


def possible_leakage_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    flagged = []

    for column in df.columns:
        lower = column.lower()
        if column == target_column:
            continue
        if any(keyword in lower for keyword in LEAKAGE_KEYWORDS):
            flagged.append(column)

    return flagged
```

### Practice

Run the leakage scan.

For each flagged column, decide:

- Safe
- Suspicious
- Must remove

## Module 8 - Create a Data Quality Report

Add:

```python
def build_quality_report(data_path: str) -> dict:
    df = pd.read_csv(data_path)

    report = {
        "data_path": data_path,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "missing_required_columns": check_required_columns(df, REQUIRED_COLUMNS),
        "missing_profile": missing_profile(df),
        "duplicate_rows": duplicate_count(df),
        "range_violations": range_violations(df, RANGE_RULES),
        "category_violations": category_violations(df, CATEGORY_RULES),
        "possible_leakage_columns": possible_leakage_columns(df, "TARGET"),
    }

    return report
```

Define rules near the top:

```python
RANGE_RULES = {
    "TARGET": (0, 1),
}

CATEGORY_RULES = {}
```

Adjust for your project.

## Module 9 - Save Markdown and JSON Reports

Add:

```python
def write_reports(report: dict) -> None:
    out_dir = ROOT / "monitoring"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "data_quality_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Data Quality Report",
        "",
        f"- Data path: `{report['data_path']}`",
        f"- Rows: {report['row_count']}",
        f"- Columns: {report['column_count']}",
        "",
        "## Required Columns",
        "",
        f"Missing required columns: {report['missing_required_columns'] or 'None'}",
        "",
        "## Duplicate Rows",
        "",
        f"Duplicate full rows: {report['duplicate_rows']}",
        "",
        "## Range Violations",
        "",
    ]

    for column, count in report["range_violations"].items():
        lines.append(f"- {column}: {count}")

    lines.extend(["", "## Possible Leakage Columns", ""])
    if report["possible_leakage_columns"]:
        for column in report["possible_leakage_columns"]:
            lines.append(f"- {column}")
    else:
        lines.append("- None flagged")

    (out_dir / "data_quality_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
```

Add:

```python
def main() -> None:
    report = build_quality_report("data/application_train.csv")
    write_reports(report)
    print("Data quality report written.")


if __name__ == "__main__":
    main()
```

### Exercise

Run:

```bash
python src/validate_data.py
```

Confirm:

```text
monitoring/data_quality_report.md
monitoring/data_quality_report.json
```

exist.

## Module 10 - Add Data Quality Tests

Create `tests/test_data_quality.py`:

```python
from pathlib import Path

import pandas as pd

from src.validate_data import check_required_columns, range_violations

ROOT = Path(__file__).resolve().parents[1]


def test_required_columns_present():
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "TARGET": [0, 1],
    })

    assert check_required_columns(df, ["SK_ID_CURR", "TARGET"]) == []


def test_range_violations_detect_bad_values():
    df = pd.DataFrame({"TARGET": [0, 1, 2]})

    result = range_violations(df, {"TARGET": (0, 1)})

    assert result["TARGET"] == 1
```

Run:

```bash
pytest tests/ -q
```

### Practice

Add one test for:

- Missing required column
- Duplicate key
- Category violation

## Module 11 - Add README Documentation

Add:

```markdown
## Data Quality

This project includes data validation checks for:

- Required columns
- Missing values
- Duplicate records
- Numeric range violations
- Invalid categorical values
- Possible target leakage columns

Run:

```bash
python src/validate_data.py
```

Outputs:

- `monitoring/data_quality_report.md`
- `monitoring/data_quality_report.json`
```

### Exercise

Add the data quality section to one project README.

## Module 12 - Explain Data Quality in Interviews

Use this answer:

```text
Before training the model, I added validation checks for required columns, missing values, duplicates, numeric ranges, categorical values, and possible leakage columns. The goal is to catch data problems early instead of letting bad data silently affect model performance. The validation script writes both JSON and Markdown reports, and I added tests for the validation logic.
```

### Practice

Answer:

1. What data quality issue would worry you most?
2. What is target leakage?
3. Why are missing values not always bad?
4. Why should validation run before training?
5. What should happen if required columns are missing?

## Final Assignment

Add data quality validation to one project.

Required:

- `src/validate_data.py`
- Required column checks
- Missing value profile
- Duplicate checks
- Numeric range checks
- Category checks
- Leakage scan
- Markdown report
- JSON report
- At least three data quality tests
- README data quality section

Optional:

- Great Expectations
- Pandera schema
- Data quality dashboard
- CI step that runs validation on a sample file

## Rubric

| Area | Strong Submission |
| --- | --- |
| Schema | Required columns are clearly defined |
| Missingness | Missing-value profile is generated |
| Duplicates | Full-row and key duplicates are checked |
| Ranges | Numeric rules catch impossible values |
| Categories | Invalid labels or statuses are detected |
| Leakage | Potential leakage columns are reviewed |
| Reports | Markdown and JSON reports are created |
| Tests | Validation logic has automated tests |
| README | Data quality workflow is documented |

## Interview Questions

Practice answering:

1. How do you validate a dataset before modeling?
2. What is target leakage?
3. How do you handle missing values?
4. How do you detect duplicate records?
5. What range checks would you add for loan data?
6. What range checks would you add for taxi data?
7. Why save a data quality report?
8. How would validation fit into CI?
9. What should block model training?
10. How would you explain a data quality failure to a stakeholder?

## Completion Checklist

You are done when:

- Validation script exists
- Data quality report is generated
- Required columns are checked
- Missing values are profiled
- Duplicates are checked
- Numeric ranges are checked
- Categories are checked
- Leakage scan runs
- Tests pass
- README explains validation

