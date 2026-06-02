# Lesson 10 - Production Readiness, CI, and Deployment Evidence

In this lesson, you will take a completed data project and make it easier for another person to trust, run, test, and review.

This is the lesson that moves a project from:

```text
It works on my laptop.
```

to:

```text
This project has tests, CI, Docker packaging, environment documentation, health checks, and deployment evidence.
```

You can apply this lesson to any earlier project:

- Credit Risk API
- Fraud Detection API
- Student Loan Forecasting
- MLOps SageMaker Churn and CLV
- NYC Taxi Lakehouse
- NLP Earnings Sentiment
- NoSQL Retail Analytics

## Learning Goals

By the end of this lesson, you should be able to:

- Add a production-readiness checklist to a project
- Write tests that protect the most important behavior
- Add a GitHub Actions CI workflow
- Add a Dockerfile and build the image locally
- Document environment variables safely
- Separate secrets from code
- Add health and readiness checks
- Create release notes
- Show deployment evidence without pretending a local app is production

## What Production Readiness Means

For a portfolio project, production readiness does not mean the project is running at a large company.

It means the project has the habits of professional software:

- Reproducible setup
- Automated tests
- Clean run commands
- Clear project structure
- No hardcoded secrets
- Model or data metadata
- API health checks
- Docker packaging
- CI checks
- Honest limitations

## Module 1 - Choose One Project to Upgrade

Pick one completed project.

Good choices:

- A FastAPI ML project
- A data pipeline project
- A model training project
- A database analytics project

Create a file called `production_readiness_audit.md`.

Add:

```markdown
# Production Readiness Audit

| Area | Current Status | Needs Work |
| --- | --- | --- |
| README setup commands |  |  |
| Tests |  |  |
| Dockerfile |  |  |
| GitHub Actions CI |  |  |
| Environment variables |  |  |
| Health checks |  |  |
| Model metadata |  |  |
| Data source documentation |  |  |
| Limitations |  |  |
| Release notes |  |  |
```

### Practice

Mark each row:

- Good
- Missing
- Needs improvement

Then choose the top three improvements.

## Module 2 - Add a Reliable Project Setup Section

Your README should let someone run the project without guessing.

Use this format:

```markdown
## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/download_data.py
python src/train.py
pytest tests/ -q
uvicorn api.app:app --reload
```
```

If the project does not have an API, replace the final command with the correct run command.

Add expected outputs:

```markdown
Expected outputs:

- `data/processed/`
- `models/model.pkl`
- `models/model_metadata.json`
- `monitoring/model_card.md`
```

### Exercise

Update one project README so a new student can run it from a fresh clone.

## Module 3 - Write Tests That Matter

Tests should protect important behavior.

For ML and data projects, useful tests include:

- Dataset has required columns
- Feature engineering returns expected columns
- Model metadata exists
- Prediction output follows a contract
- API health endpoint works
- API prediction endpoint returns valid values
- Pipeline creates expected output files

Example test file:

```python
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_has_required_columns():
    path = ROOT / "data" / "financial_phrasebank.csv"
    assert path.exists()

    df = pd.read_csv(path)
    assert {"text", "label", "label_name"}.issubset(df.columns)


def test_model_metadata_exists():
    path = ROOT / "models" / "model_metadata.json"
    assert path.exists()
```

API test example:

```python
from fastapi.testclient import TestClient

from api.app import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

### Practice

Add at least five tests:

- Two data tests
- One model or pipeline test
- One prediction contract test
- One API or output test

Run:

```bash
pytest tests/ -q
```

## Module 4 - Add Health and Readiness Checks

For API projects, add two endpoints:

```text
/healthz
/readyz
```

Health means the service is alive.

Readiness means the service can actually do its job.

Example:

```python
@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/readyz")
def ready():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifact not found")

    return {"status": "ready"}
```

### Why This Matters

In deployment, a server can be running but not ready. For example, the app might start successfully but fail because the model file is missing.

### Exercise

Add or verify:

- `/healthz`
- `/readyz`
- A test for each endpoint

## Module 5 - Add Model Metadata

A model artifact alone is not enough.

Add metadata:

```json
{
  "model_name": "credit_risk_classifier",
  "trained_at": "2026-06-02T12:00:00Z",
  "dataset": "Home Credit Default Risk",
  "metrics": {
    "test_auc": 0.86,
    "recall_at_threshold": 0.72
  },
  "features": [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "bureau_delinquency_count"
  ]
}
```

Save it as:

```text
models/model_metadata.json
```

### Practice

Create or improve `model_metadata.json`.

Include:

- Model name
- Dataset source
- Training date
- Metrics
- Feature list or feature count
- Intended use
- Limitations

## Module 6 - Add a Dockerfile

Docker makes the project easier to run in a clean environment.

Example for a FastAPI project:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build:

```bash
docker build -t my-project-api .
```

Run:

```bash
docker run --rm -p 8000:8000 my-project-api
```

### Exercise

Add Docker to one project and document:

- Build command
- Run command
- API URL or output location

## Module 7 - Document Environment Variables

Never commit secrets.

Do not commit:

- API keys
- Database passwords
- AWS secret keys
- Private tokens

Create `.env.example`:

```text
APP_ENV=local
MODEL_PATH=models/model.pkl
LOG_LEVEL=INFO
DATABASE_URL=postgresql://user:password@localhost:5432/database_name
```

Add `.env` to `.gitignore`:

```text
.env
```

### Practice

Add `.env.example` to one project.

Then answer:

1. Why do we commit `.env.example`?
2. Why do we not commit `.env`?
3. What secrets could appear in a data project?

## Module 8 - Add GitHub Actions CI

CI means continuous integration. It runs checks automatically when code is pushed.

Create:

```text
.github/workflows/ci.yml
```

Example:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run tests
        run: |
          pytest tests/ -q
```

If your tests need large data files, you have two options:

- Use a small test fixture
- Make CI run only lightweight tests

### Exercise

Add CI to one project.

Then push and check:

- Did GitHub Actions start?
- Did the test job pass?
- If it failed, what command failed?

## Module 9 - Add a Docker Build Check in CI

You can also test whether the Docker image builds.

Extend `ci.yml`:

```yaml
  docker:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Build Docker image
        run: |
          docker build -t portfolio-project .
```

### Practice

Add Docker build CI only if:

- Your project has a Dockerfile
- Docker build does not require private secrets
- Docker build does not download huge datasets every time

## Module 10 - Add Logging Basics

Logs help you understand what the app is doing.

Simple API logging:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}',
)

logger = logging.getLogger("project_api")
```

Use:

```python
logger.info("Model loaded successfully")
logger.warning("Input contained missing optional fields")
logger.error("Prediction failed")
```

### What Not to Log

Do not log:

- Passwords
- API keys
- Full Social Security numbers
- Private customer records
- Sensitive financial details

### Exercise

Add logging to one API or pipeline script.

Log:

- App start
- Model loading
- Prediction or pipeline completion
- Error conditions

## Module 11 - Add Release Notes

Create:

```text
CHANGELOG.md
```

Add:

```markdown
# Changelog

## 1.0.0

- Added reproducible data ingestion
- Added model training pipeline
- Added FastAPI scoring endpoints
- Added tests for data, prediction, and API behavior
- Added Docker packaging
- Added model metadata and governance notes
```

### Practice

Create a first release note for one project.

Use honest language:

```text
Local portfolio release
```

is better than:

```text
Enterprise production release
```

## Module 12 - Add Deployment Evidence

Deployment evidence can include:

- API docs screenshot
- Docker build screenshot
- GitHub Actions passing screenshot
- Live URL if deployed
- Terminal output showing tests passed

In your README:

```markdown
## Deployment Evidence

- Docker image builds locally
- GitHub Actions CI runs tests on push
- FastAPI docs available at `/docs`
- Health endpoint available at `/healthz`
```

If you deploy to cloud, be specific:

```markdown
This project was tested locally with Docker and structured for cloud deployment.
```

or:

```markdown
This project was deployed to AWS EC2 for demonstration with Docker.
```

Do not say a project is deployed if it is not.

### Exercise

Add one screenshot or command output summary to your README.

## Module 13 - Run the Final Review

Create:

```text
production_review.md
```

Add:

```markdown
# Production Review

## What Works

- Data ingestion runs
- Training runs
- Tests pass
- API starts
- Docker image builds

## Known Limitations

- Model is trained on historical public data
- No live monitoring dashboard yet
- No authentication on local API

## Next Improvements

- Add drift monitoring
- Add authentication
- Deploy behind managed cloud service
- Add scheduled retraining
```

### Practice

Fill out this review honestly for one project.

## Final Assignment

Upgrade one project with production-readiness artifacts.

Required:

- README quickstart
- At least five tests
- Health and readiness checks if the project has an API
- Model or pipeline metadata
- Dockerfile
- `.env.example`
- GitHub Actions CI
- Changelog
- Production review document

Optional:

- Docker build check in CI
- API logging
- Screenshots
- Cloud deployment
- Drift monitoring

## Rubric

| Area | Strong Submission |
| --- | --- |
| Reproducibility | A new user can install and run the project |
| Testing | Tests protect data, model, API, or pipeline contracts |
| CI | GitHub Actions runs meaningful checks |
| Docker | Docker image builds and run command is documented |
| Secrets | `.env.example` exists and real secrets are not committed |
| Metadata | Model or pipeline metadata explains what was built |
| Honesty | README clearly says what is local, simulated, or deployed |
| Reviewability | Recruiter or engineer can inspect the project quickly |

## Interview Questions

Practice answering:

1. What makes this project production-style?
2. What tests did you add and why?
3. What does CI check?
4. What is the difference between health and readiness?
5. How does Docker help?
6. How do you keep secrets out of GitHub?
7. What would you monitor after deployment?
8. What would fail if the model artifact were missing?
9. How would you deploy this to AWS?
10. What production features are still missing?

## Completion Checklist

You are done when:

- `pytest tests/ -q` passes locally
- Docker image builds locally
- README includes setup and run commands
- `.env.example` exists
- `.env` is ignored
- CI workflow exists
- GitHub Actions passes or failure is understood
- Changelog exists
- Production review exists
- Claims in README match what the project actually does

