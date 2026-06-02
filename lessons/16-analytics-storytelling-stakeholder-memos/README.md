# Lesson 16 - Analytics Storytelling and Stakeholder Memos

In this lesson, you will learn how to turn technical project results into clear business communication.

A model can be accurate and still fail if nobody understands what decision it supports. A data scientist must explain results, tradeoffs, risks, and recommendations in language that a stakeholder can use.

## Learning Goals

By the end of this lesson, you should be able to:

- Write a one-page stakeholder memo
- Explain model metrics in business language
- Separate findings from recommendations
- Communicate uncertainty and limitations
- Create an executive summary
- Build a simple dashboard narrative
- Present a project without overloading the audience

## What You Are Building

You will create a communication package for one project.

Final artifacts:

- `communication/executive_summary.md`
- `communication/stakeholder_memo.md`
- `communication/metric_glossary.md`
- `communication/dashboard_narrative.md`
- `communication/presentation_outline.md`

## Module 1 - Choose a Project and Audience

Pick one completed project.

Examples:

- Credit Risk API for lending managers
- Fraud Detection API for risk operations
- Student Loan Forecasting for portfolio strategy
- NYC Taxi Demand Forecasting for dispatch planning
- NLP Earnings Sentiment for financial analysts
- NoSQL Retail Analytics for ecommerce managers

Create:

```text
communication/
  executive_summary.md
  stakeholder_memo.md
  metric_glossary.md
  dashboard_narrative.md
  presentation_outline.md
```

### Practice

Answer:

1. Who is the stakeholder?
2. What decision do they need to make?
3. What output from your project helps that decision?
4. What risk should they understand before using the output?

## Module 2 - Write the Executive Summary

Create `communication/executive_summary.md`:

```markdown
# Executive Summary

## Project

Name of project.

## Business Question

What decision or workflow does this project support?

## Key Finding

State the most important result in two or three sentences.

## Recommendation

What should the stakeholder do next?

## Evidence

- Metric 1
- Metric 2
- Output 1

## Limitations

- Limitation 1
- Limitation 2
```

### Example

```text
The credit risk model ranks applicants by relative default risk using application, bureau, and previous-loan data. The model should be used to prioritize analyst review, not to make final automated approval decisions. The next step is to validate threshold choices with business cost assumptions and monitor drift before production use.
```

### Exercise

Write an executive summary for one project in fewer than 200 words.

## Module 3 - Translate Metrics

Technical metrics need plain-language interpretation.

Create `communication/metric_glossary.md`:

```markdown
# Metric Glossary

| Metric | Technical Meaning | Stakeholder Meaning |
| --- | --- | --- |
| ROC-AUC | Measures ranking quality across thresholds | How well the model separates higher-risk from lower-risk cases |
| Recall | Percent of actual positive cases identified | How many defaults or fraud cases the model catches |
| Precision | Percent of flagged cases that are true positives | How often model alerts are correct |
| Macro F1 | Average class performance across labels | Whether the model works across all classes, not only the largest class |
| MAE | Average absolute forecast error | Typical size of forecast miss |
```

### Practice

Add five metrics from your project.

For each one, write:

- Technical meaning
- Business meaning
- One limitation

## Module 4 - Write a Stakeholder Memo

Create `communication/stakeholder_memo.md`:

```markdown
# Stakeholder Memo

## To

Stakeholder role.

## From

Your name.

## Subject

Decision supported by the analysis.

## Context

Why this problem matters.

## Analysis

What data and method were used.

## Findings

1. Finding one.
2. Finding two.
3. Finding three.

## Recommendation

What action should be taken.

## Risks and Limitations

What should not be overinterpreted.

## Next Steps

1. Step one.
2. Step two.
3. Step three.
```

### Writing Rule

Do not begin with algorithms. Begin with the decision.

Weak:

```text
I used Random Forest and XGBoost.
```

Stronger:

```text
The model helps prioritize which applications need closer review by ranking relative default risk.
```

### Exercise

Write a stakeholder memo for one project.

Keep it under one page.

## Module 5 - Separate Findings and Recommendations

A finding is what the analysis shows.

A recommendation is what someone should do.

Example:

```text
Finding: High-risk applicants have higher prior delinquency counts and lower external source scores.
Recommendation: Use the model score as an analyst triage tool and review threshold choices with risk policy stakeholders.
```

### Practice

Write three findings and three recommendations.

Use this table:

```markdown
| Finding | Recommendation | Evidence |
| --- | --- | --- |
|  |  |  |
```

## Module 6 - Communicate Uncertainty

Good analysts do not hide uncertainty.

Useful phrases:

- The model suggests
- The current evidence indicates
- This should be validated with
- This result is limited by
- This should not be interpreted as

Avoid:

- The model proves
- Guaranteed
- Always
- Perfect
- Production-ready without validation

### Exercise

Rewrite these sentences:

```text
The model proves who will default.
```

```text
The forecast will predict exact taxi demand.
```

```text
The NLP model knows whether a stock will go up.
```

Make each one honest and professional.

## Module 7 - Build a Dashboard Narrative

Create `communication/dashboard_narrative.md`:

```markdown
# Dashboard Narrative

## Dashboard Purpose

What decision does the dashboard support?

## Primary KPI

What metric should the stakeholder look at first?

## Supporting Metrics

- Metric 1
- Metric 2
- Metric 3

## How to Read the Dashboard

Explain the order in which a stakeholder should inspect the visuals.

## Action Triggers

| Signal | Suggested Action |
| --- | --- |
| High-risk rate rises | Review threshold and recent data drift |
| Forecast error increases | Inspect recent demand patterns and retrain candidate model |
| Negative sentiment share rises | Review top negative sentences before analyst decision |
```

### Practice

Write a dashboard narrative for one project.

Even if the project does not have a dashboard, describe what the dashboard should show.

## Module 8 - Prepare a Five-Slide Presentation

Create `communication/presentation_outline.md`:

```markdown
# Presentation Outline

## Slide 1 - Business Problem

What decision or workflow needed support?

## Slide 2 - Data and Method

What real data was used and what approach was taken?

## Slide 3 - Results

What metrics or outputs matter most?

## Slide 4 - Recommendation

What should the stakeholder do next?

## Slide 5 - Risks and Next Steps

What are the limitations and follow-up actions?
```

### Practice

Write three bullets per slide.

Do not put code on these slides unless the audience is technical.

## Module 9 - Adjust for Audience

Different audiences need different explanations.

| Audience | Emphasize | Avoid |
| --- | --- | --- |
| Recruiter | Business problem, tools, results | Long formulas |
| Data scientist | Features, metrics, validation | Vague claims |
| Executive | Decision, risk, recommendation | Implementation details |
| Engineer | Architecture, testing, deployment | Unsupported business claims |

### Exercise

Explain the same project in four versions:

- Recruiter version
- Data scientist version
- Executive version
- Engineer version

Each version should be five sentences or fewer.

## Module 10 - Add Communication Files to README

Add:

```markdown
## Stakeholder Communication

This project includes communication artifacts for nontechnical review:

- Executive summary
- Stakeholder memo
- Metric glossary
- Dashboard narrative
- Presentation outline
```

### Practice

Link the communication files from your README.

## Final Assignment

Create a stakeholder communication package for one project.

Required:

- Executive summary
- Metric glossary
- Stakeholder memo
- Findings and recommendations table
- Dashboard narrative
- Five-slide presentation outline
- README links

Optional:

- One-page PDF memo
- Dashboard screenshot
- Recorded two-minute project presentation
- Before-and-after metric explanation

## Rubric

| Area | Strong Submission |
| --- | --- |
| Audience Fit | Language matches the stakeholder |
| Business Framing | Starts with decision, not algorithm |
| Evidence | Recommendations are tied to project outputs |
| Honesty | Limitations and uncertainty are clear |
| Clarity | Memo is concise and easy to scan |
| Portfolio Value | Communication artifacts strengthen the repo |

## Interview Questions

Practice answering:

1. How would you explain this project to an executive?
2. What decision does your model support?
3. What metric matters most to the business?
4. What should stakeholders not do with this model?
5. What recommendation comes from the analysis?
6. How would you communicate uncertainty?
7. What would be on your dashboard?
8. How would your explanation change for a technical audience?
9. What limitation matters most?
10. What is the next business step?

## Completion Checklist

You are done when:

- Communication folder exists
- Executive summary is under 200 words
- Stakeholder memo is under one page
- Metric glossary explains technical and business meaning
- Findings and recommendations are separated
- Dashboard narrative exists
- Presentation outline exists
- README links the communication artifacts
- You can explain the project without starting with code

