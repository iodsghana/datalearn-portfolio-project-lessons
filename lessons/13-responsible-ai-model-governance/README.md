# Lesson 13 - Responsible AI and Model Governance

In this lesson, you will learn how to add responsible AI and model governance documentation to a data science project.

Strong data projects are not only about accuracy. They also need clear rules for intended use, limitations, privacy, bias, monitoring, and human oversight.

This is especially important for projects involving:

- Credit risk
- Fraud detection
- Student loans
- Churn and customer value
- Financial NLP
- Forecasting used for operations or staffing

## Learning Goals

By the end of this lesson, you should be able to:

- Explain responsible AI in plain language
- Write intended-use and prohibited-use statements
- Identify sensitive features and proxy variables
- Create a model risk register
- Add bias and subgroup evaluation checks
- Document privacy and data handling rules
- Write an approval checklist
- Create governance-ready README sections
- Answer interview questions about model risk

## What Responsible AI Means

Responsible AI means building models that are:

- Useful
- Documented
- Tested
- Monitored
- Fairly evaluated
- Privacy-aware
- Honest about limitations
- Used with appropriate human oversight

For a portfolio project, you are not claiming the model is approved for real business use. You are showing that you understand the professional responsibilities around ML systems.

## Module 1 - Choose a Model Project

Pick one project:

- Credit Risk API
- Fraud Detection API
- Student Loan Forecasting
- Churn and CLV
- NLP Earnings Sentiment

Create:

```text
governance/
  intended_use.md
  risk_register.md
  bias_evaluation.md
  privacy_review.md
  approval_checklist.md
```

### Practice

Answer:

1. Who might use the model?
2. What decision could the model influence?
3. What could go wrong if the model is wrong?
4. Should a human review the model output before action is taken?

## Module 2 - Write Intended Use

Create `governance/intended_use.md`:

```markdown
# Intended Use

## Model Purpose

This model is designed to support analytical review by estimating relative risk or likelihood for a specific business workflow.

## Intended Users

- Data scientists
- Analysts
- Risk or operations teams
- Portfolio reviewers

## Intended Decisions

The model output may be used to prioritize review, compare relative risk, or support exploratory analysis.

## Not Intended For

- Fully automated approval or denial decisions
- Legal, medical, or financial advice
- Decisions without human review
- Production use without fresh validation and monitoring
```

### Exercise

Rewrite the template for your selected project.

Make it specific.

Example:

```text
The credit risk model estimates relative loan default probability for portfolio demonstration and analyst review workflows.
```

## Module 3 - Define Prohibited Uses

A strong governance document says what the model should not do.

Add:

```markdown
## Prohibited Uses

This model should not be used to:

- Make final decisions about credit, employment, housing, insurance, or education access
- Replace human review
- Score people using private or unauthorized data
- Make decisions on protected classes
- Provide investment or legal advice
- Operate in production without monitoring and approval
```

### Practice

For your project, list five prohibited uses.

Explain why each one is risky.

## Module 4 - Identify Sensitive Features

Sensitive features may include:

- Age
- Sex or gender
- Race or ethnicity
- Disability
- Marital status
- Location
- Education
- Income
- Nationality

Proxy features may indirectly reveal sensitive information.

Examples:

- ZIP code can proxy income or race
- Employment type can proxy socioeconomic status
- School type can proxy income or geography
- Text content can contain demographic signals

Create a table:

```markdown
| Feature | Sensitive or Proxy? | Risk | Action |
| --- | --- | --- | --- |
| Age | Sensitive | Could create age-based bias | Evaluate subgroup performance |
| ZIP code | Proxy | Could proxy neighborhood demographics | Consider removal or monitoring |
| Income | Sensitive financial variable | May amplify inequality | Use only with documented purpose |
```

### Exercise

Identify at least five sensitive or proxy features in your project.

If your dataset does not contain obvious sensitive features, list possible proxies.

## Module 5 - Create a Risk Register

Create `governance/risk_register.md`:

```markdown
# Model Risk Register

| Risk | Severity | Likelihood | Control |
| --- | --- | --- | --- |
| Model outputs used as final decisions | High | Medium | Document human review requirement |
| Data drift reduces reliability | Medium | Medium | Add drift monitoring |
| Missing values affect predictions | Medium | High | Validate input schema and monitor missing rates |
| Bias across subgroups | High | Unknown | Evaluate subgroup metrics when labels are available |
| Users misunderstand probabilities | Medium | Medium | Provide clear API response labels and documentation |
```

### Severity

Use:

- Low
- Medium
- High

### Practice

Add at least seven risks for your selected project.

Include:

- Data risk
- Model risk
- API risk
- User misunderstanding risk
- Monitoring risk
- Privacy risk
- Deployment risk

## Module 6 - Add Bias Evaluation

Bias evaluation means checking whether model performance differs across groups.

If your project has a subgroup column, such as age band or customer segment, you can calculate metrics by group.

Example:

```python
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score


def subgroup_metrics(df: pd.DataFrame, group_col: str, y_col: str, score_col: str) -> pd.DataFrame:
    rows = []

    for group, part in df.groupby(group_col):
        if part[y_col].nunique() < 2:
            continue

        rows.append({
            "group": group,
            "rows": len(part),
            "auc": roc_auc_score(part[y_col], part[score_col]),
            "recall_at_50pct": recall_score(part[y_col], part[score_col] >= 0.5),
            "avg_score": part[score_col].mean(),
        })

    return pd.DataFrame(rows)
```

### Exercise

Create one subgroup.

Examples:

- Age bands
- Income bands
- Loan amount bands
- Region
- Customer segment
- Text length band for NLP

Then calculate at least one metric by subgroup.

## Module 7 - Write Bias Evaluation Notes

Create `governance/bias_evaluation.md`:

```markdown
# Bias Evaluation

## Subgroups Reviewed

- Group 1
- Group 2
- Group 3

## Metrics Reviewed

- Row count
- Average prediction
- Recall
- AUC or macro F1

## Findings

Describe any performance differences.

## Limitations

- Public datasets may not contain all sensitive attributes.
- Some features may act as proxies.
- Subgroup metrics can be unstable for small groups.
- Portfolio evaluation does not replace formal fairness review.

## Actions

- Monitor subgroup performance when labels are available.
- Avoid final automated decisions.
- Require human review for high-impact use cases.
```

### Practice

Write a bias evaluation note even if your dataset lacks sensitive features.

Explain what you can and cannot evaluate.

## Module 8 - Add Privacy Review

Create `governance/privacy_review.md`:

```markdown
# Privacy Review

## Data Source

Name the dataset and link to the public source.

## Personal Data

Describe whether the dataset contains direct identifiers, indirect identifiers, or anonymized records.

## Data Handling Rules

- Do not commit private data.
- Do not log sensitive raw records.
- Do not expose private identifiers in API responses.
- Use `.env.example` instead of committing secrets.
- Store only necessary model artifacts.

## API Privacy Controls

- Limit input size.
- Validate request schemas.
- Avoid logging full request bodies.
- Return only necessary prediction outputs.
```

### Exercise

Identify:

1. What data is safe to commit?
2. What data should never be committed?
3. What should not appear in logs?
4. What should not appear in API responses?

## Module 9 - Update the Model Card

Add these sections to your model card:

```markdown
## Intended Use

Describe supported use cases.

## Prohibited Use

Describe unsupported or high-risk use cases.

## Users

Describe expected users.

## Limitations

Describe data, modeling, and deployment limitations.

## Ethical Considerations

Describe fairness, privacy, and human review considerations.

## Monitoring

Describe drift, performance, and prediction monitoring.
```

### Practice

Update one existing model card.

If your project does not have one, create:

```text
monitoring/model_card.md
```

## Module 10 - Create an Approval Checklist

Create `governance/approval_checklist.md`:

```markdown
# Model Approval Checklist

- [ ] Real data source documented
- [ ] Intended use documented
- [ ] Prohibited use documented
- [ ] Metrics reported on held-out data
- [ ] Known limitations documented
- [ ] Sensitive/proxy features reviewed
- [ ] Bias or subgroup evaluation completed where possible
- [ ] Privacy review completed
- [ ] Monitoring plan created
- [ ] Human review requirement documented
- [ ] API health and readiness checks available
- [ ] Rollback plan documented
```

### Exercise

Mark each item:

- Complete
- Needs work
- Not applicable

Then fix two `Needs work` items.

## Module 11 - Add Governance to the README

Add:

```markdown
## Governance

This project includes governance documentation for portfolio review:

- Intended use and prohibited use
- Model risk register
- Bias evaluation notes
- Privacy review
- Model approval checklist

The model is intended for analytical support and portfolio demonstration. It should not be used for final automated decisions without fresh validation, monitoring, human review, and formal approval.
```

### Practice

Add links to the governance files from your README.

## Module 12 - Explain Governance in Interviews

Use this answer pattern:

```text
I documented intended use and prohibited use so the model output is not misunderstood. I also created a risk register covering data quality, drift, privacy, and user misuse. Where possible, I evaluated subgroup performance, and I documented limitations because public datasets may not contain enough information for a full fairness review. In production, I would require monitoring, human review, access control, and periodic model validation.
```

### Practice

Answer:

1. What is the highest-risk misuse of your model?
2. What sensitive or proxy features might matter?
3. What would you monitor for fairness?
4. What would require human review?
5. What would block production approval?

## Final Assignment

Add a governance package to one project.

Required:

- `governance/intended_use.md`
- `governance/risk_register.md`
- `governance/bias_evaluation.md`
- `governance/privacy_review.md`
- `governance/approval_checklist.md`
- Updated model card
- README governance section
- At least one subgroup or proxy-feature analysis

Optional:

- Fairness metric chart
- Subgroup performance test
- Approval gate in training script
- Risk severity scoring
- Human review workflow diagram

## Rubric

| Area | Strong Submission |
| --- | --- |
| Intended Use | Specific and realistic |
| Prohibited Use | High-risk misuse is clearly blocked |
| Risk Register | Covers data, model, API, privacy, and user risks |
| Bias Review | Evaluates subgroups or documents why full review is limited |
| Privacy Review | Explains data handling and logging controls |
| Model Card | Includes limitations, ethics, and monitoring |
| README | Governance files are linked clearly |
| Interview Readiness | Student can explain governance without overclaiming |

## Interview Questions

Practice answering:

1. What is responsible AI?
2. What is the intended use of your model?
3. What should your model not be used for?
4. What sensitive features or proxies exist?
5. How did you check for bias?
6. What are the limits of your fairness evaluation?
7. What privacy risks exist?
8. What would require human review?
9. What would block production deployment?
10. How would governance change in a real company?

## Completion Checklist

You are done when:

- Governance folder exists
- Intended use is documented
- Prohibited use is documented
- Sensitive/proxy features are reviewed
- Risk register exists
- Bias evaluation exists
- Privacy review exists
- Approval checklist exists
- Model card is updated
- README links governance files
- You can explain model risk clearly

