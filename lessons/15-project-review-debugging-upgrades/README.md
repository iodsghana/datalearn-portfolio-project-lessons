# Lesson 15 - Project Review, Debugging, and Targeted Upgrades

In this lesson, you will learn how to review an existing data project like a professional engineer.

This is a high-value skill because real work often starts with an existing repository. You need to inspect it, understand it, run it, identify risks, and improve it without breaking unrelated parts.

## Learning Goals

By the end of this lesson, you should be able to:

- Review a project repository systematically
- Identify missing files, broken commands, and unclear documentation
- Run tests and interpret failures
- Prioritize improvements by risk and impact
- Make targeted upgrades instead of random refactors
- Write a clear review report
- Explain your upgrade decisions in an interview

## What You Are Building

You will create a review package for one existing project.

Final artifacts:

- `review/project_review.md`
- `review/test_results.md`
- `review/upgrade_plan.md`
- `review/final_summary.md`
- At least three targeted project improvements

## Module 1 - Choose a Project to Review

Pick one project you already built.

Good choices:

- Credit Risk API
- Fraud Detection API
- Student Loan Forecasting
- NYC Taxi Lakehouse
- NLP Earnings Sentiment
- NoSQL Retail Analytics

Create:

```text
review/
  project_review.md
  test_results.md
  upgrade_plan.md
  final_summary.md
```

### Practice

Before opening code, answer:

1. What do you expect this project to do?
2. What should the README tell you?
3. What command should run tests?
4. What would make the project hard for a reviewer?

## Module 2 - Inspect the Repository Structure

Run:

```bash
dir
```

or:

```bash
ls
```

Then inspect files:

```bash
rg --files
```

If `rg` is not installed, use:

```bash
Get-ChildItem -Recurse
```

Create `review/project_review.md`:

```markdown
# Project Review

## Repository Structure

Important folders:

- `src/`
- `api/`
- `tests/`
- `data/`
- `models/`
- `monitoring/`

## First Impressions

- What is clear:
- What is unclear:
- What appears missing:
```

### Exercise

List the five most important files in the project.

For each one, write what it does.

## Module 3 - Review the README

A strong README should answer:

- What problem does the project solve?
- What real data is used?
- How do you install dependencies?
- How do you run the project?
- How do you run tests?
- What results were achieved?
- What are the limitations?

Add this table:

```markdown
## README Review

| Item | Present? | Notes |
| --- | --- | --- |
| Problem statement |  |  |
| Dataset source |  |  |
| Setup commands |  |  |
| Run commands |  |  |
| Test commands |  |  |
| Metrics |  |  |
| API endpoints |  |  |
| Limitations |  |  |
```

### Practice

Mark each row:

- Yes
- No
- Partial

Then choose two README improvements.

## Module 4 - Run the Project

Start with setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the main workflow:

```bash
python src/download_data.py
python src/train.py
pytest tests/ -q
```

If the project has an API:

```bash
uvicorn api.app:app --reload
```

### Practice

Record what worked and what failed in `review/test_results.md`:

```markdown
# Test Results

## Commands Run

| Command | Result | Notes |
| --- | --- | --- |
| `pip install -r requirements.txt` |  |  |
| `python src/train.py` |  |  |
| `pytest tests/ -q` |  |  |

## Failures

- Failure:
- Likely cause:
- Fix:
```

## Module 5 - Diagnose Failures

When something fails, do not panic. Debug in layers.

Common failure types:

| Error Type | Common Cause |
| --- | --- |
| Missing file | Data or model artifact not created |
| Import error | Wrong path or missing dependency |
| Test failure | Behavior changed or test expectation outdated |
| API 503 | Model artifact missing |
| Bad prediction output | Preprocessing mismatch |
| Docker failure | Requirements or file paths wrong |

Debugging questions:

1. Did the command ever work?
2. What file or dependency is missing?
3. Did a previous step need to run first?
4. Is the README command accurate?
5. Is the test catching a real bug?

### Exercise

Pick one failure or weakness.

Write:

```text
Symptom:
Likely cause:
Evidence:
Fix:
How I verified:
```

## Module 6 - Prioritize Upgrades

Not every issue matters equally.

Use this priority scale:

| Priority | Meaning |
| --- | --- |
| P0 | Project cannot run |
| P1 | Core behavior broken |
| P2 | Important quality improvement |
| P3 | Nice-to-have polish |

Create `review/upgrade_plan.md`:

```markdown
# Upgrade Plan

| Priority | Issue | Proposed Fix | Verification |
| --- | --- | --- | --- |
| P1 | Tests fail because model metadata is missing | Add training step and metadata file | `pytest tests/ -q` |
| P2 | README lacks API examples | Add endpoint table and sample request | Manual review |
| P2 | No smoke test | Add `scripts/smoke_test.py` | Run smoke test |
```

### Practice

List at least five possible upgrades.

Choose three to implement.

## Module 7 - Make Targeted Improvements

Good upgrades are focused.

Examples:

- Add missing README commands
- Add tests for API health
- Add `.env.example`
- Add model metadata
- Add smoke test
- Add data validation
- Add a monitoring note
- Add Docker run instructions
- Fix broken imports

Avoid:

- Rewriting the entire project without need
- Changing unrelated files
- Deleting working code
- Adding complex tools no one asked for

### Exercise

Implement three upgrades.

For each one, document:

```text
Upgrade:
Why it matters:
Files changed:
Verification:
```

## Module 8 - Improve Test Coverage

Every project should test its most important behavior.

Add tests for:

- Data schema
- Feature engineering
- Model metadata
- Prediction output
- API health
- API readiness
- Pipeline output files

Example:

```python
def test_prediction_contract():
    from src.predict import predict

    result = predict({"feature_a": 1.0, "feature_b": 2.0})

    assert isinstance(result, dict)
    assert "prediction" in result
```

### Practice

Add one new test.

Run:

```bash
pytest tests/ -q
```

## Module 9 - Improve Documentation

Add or improve:

- Project summary
- Data source
- Metrics table
- Endpoint table
- How to run
- How to test
- Limitations

Example:

```markdown
## How to Verify

```bash
python src/download_data.py
python src/train.py
pytest tests/ -q
uvicorn api.app:app --reload
```
```

### Practice

Add a `How to Verify` section to one project README.

## Module 10 - Write the Final Summary

Create `review/final_summary.md`:

```markdown
# Final Review Summary

## What Was Reviewed

- README
- Data pipeline
- Training script
- API
- Tests
- Docker or deployment files

## Issues Found

- Issue 1
- Issue 2
- Issue 3

## Improvements Made

- Improvement 1
- Improvement 2
- Improvement 3

## Verification

- Tests passed:
- API started:
- Smoke test passed:

## Remaining Risks

- Risk 1
- Risk 2
```

### Practice

Write the final summary as if a hiring manager will read it.

Keep it honest and concise.

## Module 11 - Interview Explanation

Use this answer:

```text
I reviewed the project by first reading the README and repository structure, then running the documented setup, training, testing, and API commands. I recorded failures, prioritized them by risk, and made targeted improvements rather than rewriting unrelated code. The main upgrades were documentation clarity, test coverage, and verification commands. I confirmed the work by rerunning tests and documenting remaining risks.
```

### Practice

Answer:

1. How do you approach an unfamiliar repository?
2. How do you decide what to fix first?
3. How do you know whether a test failure is a real bug?
4. Why avoid unnecessary refactoring?
5. How do you document remaining risk?

## Final Assignment

Review and upgrade one existing project.

Required:

- `review/project_review.md`
- `review/test_results.md`
- `review/upgrade_plan.md`
- `review/final_summary.md`
- At least three targeted improvements
- At least one new or improved test
- README improvement
- Verification command output summarized

Optional:

- Smoke test
- Docker check
- CI improvement
- Monitoring note
- Governance note

## Rubric

| Area | Strong Submission |
| --- | --- |
| Review Quality | Finds real clarity, reliability, or testing gaps |
| Prioritization | Fixes high-impact issues first |
| Debugging | Uses evidence instead of guessing |
| Scope Control | Avoids unrelated rewrites |
| Verification | Reruns tests or smoke checks |
| Documentation | Final summary is clear and honest |
| Interview Readiness | Student can explain review process |

## Interview Questions

Practice answering:

1. What do you check first in a new repo?
2. How do you find the main entrypoints?
3. What makes a README useful?
4. How do you debug failing tests?
5. How do you prioritize bugs?
6. What is a smoke test?
7. Why should upgrades be targeted?
8. How do you avoid breaking user changes?
9. How do you verify a fix?
10. What risks remained after your review?

## Completion Checklist

You are done when:

- Review folder exists
- README was reviewed
- Commands were tested
- Issues were prioritized
- Three improvements were made
- At least one test was added or improved
- Final summary exists
- Remaining risks are documented
- You can explain the review process clearly

