# Lesson 17 - Technical Interview and Project Defense Lab

In this lesson, you will practice explaining and defending your portfolio projects in technical interviews.

Building the project is only half the work. You also need to explain why you made choices, what tradeoffs you considered, how you tested the system, and how you would improve it.

## Learning Goals

By the end of this lesson, you should be able to:

- Defend a portfolio project under interview pressure
- Answer SQL, Python, ML, API, and system-design questions
- Explain metrics and tradeoffs clearly
- Discuss limitations without sounding unprepared
- Use structured answers instead of rambling
- Practice with a scoring rubric

## What You Are Building

You will create an interview preparation packet:

```text
interview/
  project_defense.md
  sql_drills.md
  python_drills.md
  ml_drills.md
  system_design_drills.md
  mock_interview_scorecard.md
```

## Module 1 - Choose Three Projects to Defend

Pick three projects:

- One machine learning API
- One data engineering or database project
- One advanced project such as NLP, MLOps, forecasting, or monitoring

Create `interview/project_defense.md`:

```markdown
# Project Defense

| Project | Main Skill Signal | Why I Built It | Hardest Part |
| --- | --- | --- | --- |
| Credit Risk API | ML + finance + API | To model loan default risk with real data | Joining bureau and previous loan history |
| NYC Taxi Lakehouse | Data engineering + forecasting | To build a lakehouse-style demand pipeline | Creating reliable hourly demand marts |
| NLP Earnings Sentiment | NLP + API | To classify financial text sentiment | Aggregating sentence scores into transcript sentiment |
```

### Practice

For each project, write:

```text
The business problem:
The dataset:
The core technical approach:
The main metric:
The biggest limitation:
The next improvement:
```

## Module 2 - Use the 90-Second Project Defense

Use this answer structure:

```text
1. Problem
2. Data
3. Approach
4. Evaluation
5. Deployment or output
6. Limitation
7. Next improvement
```

Example:

```text
My credit risk project estimates relative loan default risk. I used real Home Credit application data, bureau records, and previous application history. I engineered applicant-level features, trained supervised classifiers, and evaluated ranking and threshold performance. I packaged the model behind FastAPI with health checks, model metadata, tests, and governance notes. The main limitation is that it uses historical public data, so production use would require fresh validation and drift monitoring. My next improvement would be calibration and subgroup performance review.
```

### Exercise

Write a 90-second defense for each of your three projects.

Read it out loud and cut anything that sounds like filler.

## Module 3 - SQL Interview Drills

Create `interview/sql_drills.md`.

Practice these:

```sql
-- 1. Count records by status
SELECT status, COUNT(*) AS n
FROM applications
GROUP BY status
ORDER BY n DESC;

-- 2. Find default rate by income band
SELECT
  income_band,
  AVG(CASE WHEN default_flag = 1 THEN 1.0 ELSE 0.0 END) AS default_rate,
  COUNT(*) AS applicants
FROM loan_applications
GROUP BY income_band
ORDER BY default_rate DESC;

-- 3. Rank zones by hourly demand
SELECT
  pickup_zone,
  pickup_hour,
  trip_count,
  RANK() OVER (PARTITION BY pickup_hour ORDER BY trip_count DESC) AS demand_rank
FROM hourly_taxi_demand;
```

### Practice Questions

Answer:

1. What is the difference between `WHERE` and `HAVING`?
2. What is a window function?
3. How would you find duplicate records?
4. How would you calculate a rolling seven-day average?
5. How would you join application data to bureau history?

## Module 4 - Python Interview Drills

Create `interview/python_drills.md`.

Practice:

```python
def missing_rate_by_column(df):
    return df.isna().mean().sort_values(ascending=False)
```

```python
def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0
    return numerator / denominator
```

```python
def batch_predict(rows, predict_fn):
    results = []
    for row in rows:
        results.append(predict_fn(row))
    return results
```

### Practice Questions

Answer:

1. What is the difference between a list and a tuple?
2. What is a dictionary useful for?
3. How do you handle missing values in pandas?
4. What does `groupby` do?
5. Why should file paths use `pathlib`?
6. What is the difference between `fit_transform` and `transform`?

## Module 5 - Machine Learning Interview Drills

Create `interview/ml_drills.md`.

Practice answering:

```text
Why did you choose this model?
What baseline did you compare against?
What metric did you optimize?
How did you avoid data leakage?
How did you handle class imbalance?
How would you monitor this model?
When would you retrain it?
```

Strong answer pattern:

```text
I chose [model] because [reason]. I evaluated it with [metric] because [business reason]. I protected against [risk] by [method]. In production, I would monitor [signals] and retrain only after confirming that drift or performance degradation is real.
```

### Exercise

Answer these for one classification project:

1. Why not just use accuracy?
2. What happens if classes are imbalanced?
3. What is threshold tuning?
4. What is model calibration?
5. What is data leakage?

## Module 6 - API and Deployment Drills

Practice:

```text
Why FastAPI?
What does /healthz check?
What does /readyz check?
How is the model loaded?
How would you deploy it?
What would you log?
What should not be logged?
How would you roll back a bad deployment?
```

Strong answer:

```text
I used FastAPI because it gives request validation, automatic OpenAPI docs, and clean endpoint definitions. Health checks show the service is alive, while readiness checks confirm required artifacts like the model and metadata can load. For deployment, I would containerize with Docker, run behind an authenticated gateway, monitor latency and error rates, and roll back to the previous working image if readiness or prediction checks fail.
```

### Practice

Write answers for:

1. Health versus readiness
2. Docker benefit
3. Environment variables
4. API authentication
5. Logging and privacy

## Module 7 - System Design Drill

Create `interview/system_design_drills.md`.

Prompt:

```text
Design a loan default risk scoring system.
```

Use this structure:

```text
Inputs:
Data ingestion:
Feature engineering:
Model training:
Model registry:
API serving:
Monitoring:
Human review:
Governance:
Failure handling:
```

Example:

```text
Applications and bureau records flow into a data processing pipeline. The pipeline validates schemas, creates applicant-level features, and stores model-ready data. A training job evaluates candidate models and registers the best model with metadata. A FastAPI service loads the approved model and returns default risk scores. Monitoring tracks latency, errors, drift, prediction distribution, and later performance labels. High-impact decisions require human review and governance approval.
```

### Exercise

Design one of these:

- Fraud detection scoring system
- Taxi demand forecasting service
- Earnings sentiment API
- Churn retention recommendation engine

## Module 8 - Behavioral Questions from Projects

Practice STAR:

```text
Situation:
Task:
Action:
Result:
```

Questions:

1. Tell me about a difficult project.
2. Tell me about a time you had to debug something.
3. Tell me about a time you made a tradeoff.
4. Tell me about a time you explained a technical result.
5. Tell me about a time you improved a process.

### Exercise

Write two STAR answers based on your projects.

Keep each answer under two minutes.

## Module 9 - Create a Mock Interview Scorecard

Create `interview/mock_interview_scorecard.md`:

```markdown
# Mock Interview Scorecard

| Area | Score 1-5 | Notes |
| --- | ---: | --- |
| Problem framing |  |  |
| Data understanding |  |  |
| Technical depth |  |  |
| Metric explanation |  |  |
| Deployment understanding |  |  |
| Monitoring/governance |  |  |
| Communication clarity |  |  |
| Conciseness |  |  |

## Strengths

- 

## Improvements

- 

## Next Practice Focus

- 
```

### Practice

Record yourself answering one project defense question.

Score yourself honestly.

## Module 10 - Weekly Interview Practice Plan

Use this schedule:

| Day | Practice |
| --- | --- |
| Monday | One project defense |
| Tuesday | SQL drills |
| Wednesday | ML metric questions |
| Thursday | API/deployment questions |
| Friday | System design prompt |
| Saturday | Mock interview recording |
| Sunday | Review and rewrite weak answers |

### Exercise

Create your own seven-day practice plan.

## Final Assignment

Create a complete interview packet.

Required:

- Three 90-second project defenses
- SQL drill answers
- Python drill answers
- ML drill answers
- API/deployment answers
- One system design answer
- Two STAR stories
- Mock interview scorecard
- Seven-day practice plan

## Rubric

| Area | Strong Submission |
| --- | --- |
| Project Defense | Clear, specific, and under 90 seconds |
| SQL | Answers include grouping, joins, and window functions |
| Python | Answers show practical pandas and function skills |
| ML | Explains metrics, leakage, imbalance, and monitoring |
| Deployment | Understands FastAPI, Docker, health checks, and logs |
| System Design | Covers data, model, API, monitoring, and governance |
| Communication | Answers are concise and honest |

## Interview Questions

Practice answering:

1. Tell me about your strongest project.
2. Why did you choose that dataset?
3. What was the hardest technical part?
4. How did you evaluate the model?
5. What would you monitor in production?
6. How would you deploy the API?
7. How would you handle data drift?
8. What would you improve next?
9. How would you explain the project to a business stakeholder?
10. Why should we trust the results?

## Completion Checklist

You are done when:

- Interview folder exists
- Three project defenses are written
- SQL drills are answered
- Python drills are answered
- ML questions are answered
- API/deployment questions are answered
- System design answer is written
- STAR stories are ready
- Scorecard is completed
- You have practiced out loud

