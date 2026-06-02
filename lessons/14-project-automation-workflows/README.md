# Lesson 14 - Project Automation Workflows

In this lesson, you will turn a data project into a repeatable workflow.

Many beginner projects require a reader to guess which command comes first:

```text
Should I download data first?
Train first?
Run tests first?
Start the API?
Where does the model artifact come from?
```

Professional projects remove that confusion. They provide simple commands for setup, data preparation, training, testing, serving, monitoring, and cleanup.

## Learning Goals

By the end of this lesson, you should be able to:

- Define a repeatable project workflow
- Create automation commands for common tasks
- Add a Windows-friendly task runner
- Add a Makefile-style command list
- Create smoke tests
- Document workflow commands in the README
- Explain pipeline order in interviews

## What You Are Building

You will add automation to one existing project.

Final artifacts:

- `Makefile` or `tasks.ps1`
- `scripts/smoke_test.py`
- README command table
- Workflow diagram
- Local run checklist
- Optional CI command reuse

## Module 1 - Choose a Project

Pick one project with at least three steps.

Good choices:

- Credit Risk API
- Fraud Detection API
- Student Loan Forecasting
- NYC Taxi Lakehouse
- NLP Earnings Sentiment

Create:

```text
scripts/
  smoke_test.py
tasks.ps1
```

Optional:

```text
Makefile
```

### Practice

Write the correct command order for your project:

```text
1. Create environment
2. Install dependencies
3. Download data
4. Train model
5. Run tests
6. Start API
7. Run smoke test
```

## Module 2 - Create a Command Inventory

Create `workflow.md`:

```markdown
# Project Workflow

| Step | Command | Output |
| --- | --- | --- |
| Install dependencies | `pip install -r requirements.txt` | Python packages installed |
| Download data | `python src/download_data.py` | Data files created |
| Train model | `python src/train.py` | Model artifact and metadata |
| Run tests | `pytest tests/ -q` | Test results |
| Start API | `uvicorn api.app:app --reload` | Local API |
| Smoke test | `python scripts/smoke_test.py` | Endpoint check |
```

### Exercise

Fill in the table for one project.

If a command does not apply, remove it.

## Module 3 - Add a Windows Task Script

Create `tasks.ps1`.

```powershell
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("install", "data", "train", "test", "serve", "smoke", "all")]
    [string]$Task
)

if ($Task -eq "install") {
    python -m pip install --upgrade pip
    pip install -r requirements.txt
}

if ($Task -eq "data") {
    python src/download_data.py
}

if ($Task -eq "train") {
    python src/train.py
}

if ($Task -eq "test") {
    pytest tests/ -q
}

if ($Task -eq "serve") {
    uvicorn api.app:app --reload
}

if ($Task -eq "smoke") {
    python scripts/smoke_test.py
}

if ($Task -eq "all") {
    python src/download_data.py
    python src/train.py
    pytest tests/ -q
}
```

Run:

```powershell
.\tasks.ps1 install
.\tasks.ps1 data
.\tasks.ps1 train
.\tasks.ps1 test
```

### Practice

Modify the task names for your project.

For a NoSQL project, you might use:

```text
load
aggregate
api
test
```

For a lakehouse project, you might use:

```text
download
etl
train
forecast
test
```

## Module 4 - Add a Makefile

Some developers use Makefiles to create short commands.

Create `Makefile`:

```makefile
.PHONY: install data train test serve smoke all

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

data:
	python src/download_data.py

train:
	python src/train.py

test:
	pytest tests/ -q

serve:
	uvicorn api.app:app --reload

smoke:
	python scripts/smoke_test.py

all: data train test
```

Run:

```bash
make test
```

On Windows, `make` may not be installed. That is why `tasks.ps1` is useful.

### Exercise

Add a command that builds Docker:

```makefile
docker-build:
	docker build -t portfolio-project .
```

## Module 5 - Create a Smoke Test

A smoke test is a quick check that the main system works.

For an API project, create `scripts/smoke_test.py`:

```python
from __future__ import annotations

import json
from urllib.request import Request, urlopen


BASE_URL = "http://127.0.0.1:8000"


def get_json(path: str) -> dict:
    with urlopen(f"{BASE_URL}{path}", timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(path: str, payload: dict) -> dict:
    request = Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    health = get_json("/healthz")
    print("health:", health)

    ready = get_json("/readyz")
    print("ready:", ready)

    # Replace this payload with your project's prediction example.
    prediction = post_json(
        "/v1/sentence",
        {"text": "Operating profit increased after strong demand."},
    )
    print("prediction:", prediction)


if __name__ == "__main__":
    main()
```

### Practice

Change:

- `BASE_URL`
- Prediction endpoint
- Request payload
- Expected response fields

## Module 6 - Create a Pipeline Diagram

Add this to `workflow.md`:

```text
Install dependencies
  -> download data
  -> validate data
  -> train model
  -> save metadata
  -> run tests
  -> start API
  -> smoke test
```

For a lakehouse project:

```text
Download raw data
  -> build processed fact table
  -> create marts
  -> train forecast model
  -> generate reports
  -> run tests
```

### Exercise

Draw your project workflow in plain text.

Keep it under ten steps.

## Module 7 - Add README Command Table

Add this section to your README:

```markdown
## Common Commands

| Task | Windows | Make |
| --- | --- | --- |
| Install dependencies | `.\tasks.ps1 install` | `make install` |
| Download data | `.\tasks.ps1 data` | `make data` |
| Train model | `.\tasks.ps1 train` | `make train` |
| Run tests | `.\tasks.ps1 test` | `make test` |
| Start API | `.\tasks.ps1 serve` | `make serve` |
| Smoke test | `.\tasks.ps1 smoke` | `make smoke` |
```

### Practice

Update the command table for your project.

If the project does not have an API, replace `serve` with the correct command.

## Module 8 - Reuse Commands in CI

Your CI workflow can reuse the same commands.

Example `.github/workflows/ci.yml`:

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
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: make install

      - name: Run tests
        run: make test
```

If `make` is not available or not wanted:

```yaml
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run tests
        run: |
          pytest tests/ -q
```

### Exercise

Decide whether your project should use:

- Makefile in CI
- Direct commands in CI
- PowerShell tasks only for local Windows use

Explain your choice.

## Module 9 - Add Clean Commands

Add a cleanup command to remove temporary files.

PowerShell:

```powershell
if ($Task -eq "clean") {
    Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
    Get-ChildItem -Recurse -Directory -Filter ".pytest_cache" | Remove-Item -Recurse -Force
}
```

Makefile:

```makefile
clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
```

### Practice

Add a clean command.

Be careful: cleanup commands should remove only generated cache folders, not data or source code.

## Module 10 - Add a Local Review Checklist

Create `local_review.md`:

```markdown
# Local Review Checklist

- [ ] Fresh environment created
- [ ] Dependencies installed
- [ ] Data step runs
- [ ] Training or ETL step runs
- [ ] Tests pass
- [ ] API starts if applicable
- [ ] Smoke test passes
- [ ] README commands match actual commands
- [ ] No secrets are committed
- [ ] Git status is clean before push
```

### Exercise

Run through the checklist for one project.

Fix anything that fails.

## Module 11 - Explain Automation in Interviews

Use this answer:

```text
I added task automation so the project can be run consistently by another person. Instead of relying on scattered manual commands, the repo has named tasks for data ingestion, training, testing, serving, and smoke testing. This also makes CI easier because the same commands can be reused in automated checks.
```

### Practice

Answer:

1. Why is command automation useful?
2. What does a smoke test check?
3. Why should README commands be tested?
4. How does task automation help CI?

## Final Assignment

Add workflow automation to one project.

Required:

- `workflow.md`
- `tasks.ps1`
- `scripts/smoke_test.py` if the project has an API
- README command table
- Local review checklist
- At least one automation command for tests

Optional:

- `Makefile`
- Docker build command
- CI workflow that reuses automation commands
- Clean command
- Monitoring command

## Rubric

| Area | Strong Submission |
| --- | --- |
| Workflow | Correct order of project steps is documented |
| Automation | Tasks run real project commands |
| Smoke Test | Main API or output can be checked quickly |
| README | Command table is clear and accurate |
| CI Reuse | Commands can support automated testing |
| Safety | Clean commands avoid deleting important files |
| Interview Readiness | Student can explain why automation matters |

## Interview Questions

Practice answering:

1. How do I run your project from scratch?
2. What command trains the model?
3. What command runs tests?
4. What is a smoke test?
5. What happens if the model artifact is missing?
6. How do your local commands relate to CI?
7. How would you automate retraining?
8. What generated files should be cleaned?
9. What should not be deleted by a clean command?
10. How does automation improve reproducibility?

## Completion Checklist

You are done when:

- Workflow file exists
- Task script exists
- Commands run successfully
- README command table is updated
- Smoke test exists if applicable
- Local review checklist exists
- You can explain the project workflow without guessing

