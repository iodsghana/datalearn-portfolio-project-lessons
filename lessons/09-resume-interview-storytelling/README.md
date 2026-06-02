# Lesson 09 - Resume and Interview Storytelling for Data Projects

In this lesson, you will learn how to turn completed data projects into resume bullets and interview stories.

This is not a lesson about making claims sound bigger than they are. It is a lesson about making real work easy to understand.

Many students build good projects but describe them weakly:

```text
Built a machine learning model.
```

A stronger version explains the problem, data, method, and output:

```text
Built a credit risk scoring API using real Home Credit application, bureau, and previous-loan data; engineered borrower-level risk features, evaluated default prediction performance, and served model scores through FastAPI with tests and governance documentation.
```

By the end, you will have:

- Project resume bullets
- Recruiter-friendly summaries
- Technical interview stories
- STAR responses
- A project ranking matrix
- A one-minute portfolio pitch
- A two-minute deep-dive answer for each major project

## Learning Goals

By the end of this lesson, you should be able to:

- Translate project work into strong resume bullets
- Explain a project clearly to a recruiter
- Explain the same project technically to a hiring manager
- Describe business value without exaggerating
- Answer "Tell me about a project" with structure
- Prepare for model, API, data, and system-design questions
- Connect projects to the job you want

## What Makes a Strong Project Story?

A strong project story answers five questions:

1. What problem did you solve?
2. What data did you use?
3. What did you build?
4. How did you evaluate it?
5. Why does it matter?

Use this formula:

```text
I built [system/project] using [real data] to solve [business problem].
The system includes [technical components].
I evaluated it using [metrics/tests].
The result is [usable output].
```

## Module 1 - Inventory Your Projects

Create a file called `project_story_inventory.md`.

Add this table:

```markdown
| Project | Business Problem | Real Data | Technical Work | Evaluation | Output |
| --- | --- | --- | --- | --- | --- |
| Credit Risk API | Predict default risk | Home Credit | Feature engineering, classification, API | AUC, recall, tests | Default probability API |
| NYC Taxi Lakehouse | Forecast ride demand | NYC TLC, weather | ETL, marts, forecasting | MAE/RMSE, tests | Demand forecast API |
| NLP Earnings Sentiment | Classify financial tone | Financial PhraseBank | TF-IDF, lexicon features, API | Accuracy, macro F1 | Sentence/transcript scoring |
```

Fill in your own projects.

### Practice

For each project, write one sentence:

```text
This project helps [user/company/team] do [decision/workflow] by using [data/model/system].
```

Example:

```text
This project helps lending teams rank loan applicants by default risk by using real borrower, bureau, and prior-loan data.
```

## Module 2 - Write Resume Bullets the Right Way

A strong resume bullet has four parts:

- Action verb
- Technical method
- Data or scale
- Result or output

Template:

```text
[Action verb] [system/model/pipeline] using [tools/data/methods] to [business outcome], producing [metric/artifact/output].
```

Examples:

```text
Built a production-style credit risk API using Home Credit application, bureau, and previous-loan data; engineered borrower-level risk features and served default probabilities through FastAPI with automated tests.
```

```text
Developed an NYC taxi demand lakehouse using official TLC trip data and weather enrichment; created hourly zone-demand marts, trained forecasting models, and exposed demand predictions through an API.
```

```text
Created a financial NLP sentiment API trained on Financial PhraseBank; combined TF-IDF and finance lexicon features, generated model metadata, and served sentence, transcript, and batch predictions.
```

### Weak vs Strong

Weak:

```text
Used Python for machine learning.
```

Strong:

```text
Trained and evaluated a loan default classifier in Python using real credit application data, engineered bureau and prior-loan features, and packaged predictions behind a FastAPI scoring endpoint.
```

### Practice

Write three bullets for one project:

1. A modeling bullet
2. An engineering bullet
3. A business-value bullet

Then revise each bullet to remove vague words like:

- helped
- worked on
- used
- did
- made

## Module 3 - Use Honest Metrics

Metrics make bullets stronger, but they must be supported by the project.

Good project metrics:

- ROC-AUC
- Recall
- Precision
- Macro F1
- RMSE
- MAE
- API latency
- Rows processed
- Tests passed
- Number of endpoints
- Number of engineered features

Bad or risky metrics:

- Revenue increase you did not measure
- Cost reduction you did not calculate
- "Production impact" if the project was local
- Claims copied from another project

Better wording:

```text
Processed 300K+ application records from the Home Credit dataset.
```

```text
Exposed six API endpoints for health checks, model metadata, and prediction workflows.
```

```text
Achieved 0.87 macro F1 on the held-out Financial PhraseBank test set.
```

### Exercise

For each project, write:

```text
Supported metric:
Where it appears in the repo:
How I would explain it:
```

Example:

```text
Supported metric: Test macro F1 = 0.8717
Where it appears in the repo: models/model_metadata.json and README
How I would explain it: Macro F1 averages performance across negative, neutral, and positive sentiment classes, which is useful when label counts are uneven.
```

## Module 4 - Build a One-Minute Project Pitch

A recruiter may ask:

```text
Tell me about one of your projects.
```

Do not start with code. Start with the problem.

Use this structure:

```text
The project solves [business problem].
I used [real dataset].
I built [system].
The most important technical parts were [2-3 details].
I evaluated it with [metrics/tests].
The final output was [API/dashboard/report/model].
```

Example:

```text
One project I built is a credit risk API. The business problem is helping lenders estimate default risk before approving loans. I used real Home Credit application data, bureau history, and previous-loan records. I built a feature engineering pipeline, trained classification models, evaluated risk thresholds, and served predictions through FastAPI. The final output is a scoring API with model metadata, tests, and governance notes.
```

### Practice

Write a one-minute pitch for:

- Your strongest ML project
- Your strongest data engineering project
- Your strongest NLP or analytics project

Read each one out loud. If it takes more than one minute, shorten it.

## Module 5 - Build a Technical Deep Dive

A hiring manager may ask:

```text
Walk me through the technical design.
```

Use this structure:

```text
Data source:
Data cleaning:
Feature engineering:
Model or analytics method:
Evaluation:
Deployment or output:
Testing:
Limitations:
Next improvement:
```

Example:

```text
Data source: Home Credit application_train, bureau, and previous_application files.
Data cleaning: Handled missing values, categorical encoding, and applicant-level joins.
Feature engineering: Created bureau delinquency aggregates and previous-loan refusal counts.
Model: Trained supervised classifiers to estimate default probability.
Evaluation: Compared validation metrics and reviewed threshold tradeoffs.
Deployment: Served predictions through FastAPI.
Testing: Added tests for preprocessing, prediction contracts, and API readiness.
Limitations: The model is trained on historical competition data, so production use would require fresh validation and monitoring.
Next improvement: Add drift monitoring and calibration checks.
```

### Exercise

Create a technical deep dive for two projects.

Keep each one under two minutes.

## Module 6 - Prepare STAR Stories

STAR means:

- Situation
- Task
- Action
- Result

Use it for behavioral questions.

Question:

```text
Tell me about a challenging technical project.
```

STAR answer:

```text
Situation: I wanted to build a project that showed more than notebook modeling.
Task: I needed to use real data, create a reproducible pipeline, and serve predictions.
Action: I built a credit risk API using Home Credit data, engineered applicant-level features, trained and evaluated classifiers, added FastAPI endpoints, and wrote tests.
Result: The project became a portfolio-ready ML system with model metadata, governance notes, and a working prediction API.
```

### Practice

Write STAR answers for:

1. A project where you solved a data quality problem
2. A project where you made a modeling choice
3. A project where you improved engineering structure
4. A project where you had to explain limitations

## Module 7 - Answer Model Questions

Common questions:

- Why did you choose this model?
- What baseline did you compare against?
- What metric did you use and why?
- How did you avoid data leakage?
- How did you handle class imbalance?
- What would you monitor in production?

Answer pattern:

```text
I chose [model] because [reason].
I evaluated it using [metric] because [business need].
I protected against [risk] by [method].
If this were production, I would monitor [signals].
```

Example:

```text
I used logistic regression for the NLP sentiment model because TF-IDF features work well with linear classifiers, it trains quickly, and it is easier to deploy and explain. I used macro F1 because the sentiment classes are not equally distributed, and I wanted performance across all classes rather than only the majority class.
```

### Practice

For one project, answer:

1. Why this model?
2. Why this metric?
3. What could go wrong?
4. How would you improve it?

## Module 8 - Answer Data Engineering Questions

Common questions:

- How does data flow through the project?
- What files or tables are created?
- How do you validate the data?
- What would happen if a source file changed?
- How would you scale this pipeline?
- How would you schedule it?

Answer pattern:

```text
The pipeline starts with [source].
It creates [intermediate outputs].
It validates [rules].
The final output is [mart/model/API].
To scale it, I would [scaling plan].
```

Example:

```text
The NYC taxi project starts with official TLC trip data and weather data. It builds a processed trip fact table, hourly zone-demand marts, daily summaries, and model-ready forecasting features. The pipeline validates required columns and filters invalid trips. To scale it, I would move the raw and processed layers to object storage and run transformations with Spark or a scheduled orchestration tool.
```

### Exercise

Write a data-flow answer for one project.

Use no more than eight sentences.

## Module 9 - Answer API and Deployment Questions

Common questions:

- Why did you use FastAPI?
- What does `/healthz` do?
- What does `/readyz` do?
- How is the model loaded?
- How would you deploy this?
- How would you secure it?
- How would you monitor it?

Strong answer:

```text
I used FastAPI because it gives automatic request validation, OpenAPI documentation, and clean endpoint definitions. The health endpoint checks whether the service is alive, while readiness checks whether the model artifact and metadata can load. For production, I would containerize the app, deploy it behind an authenticated gateway, log prediction metadata, and monitor latency, errors, drift, and prediction distributions.
```

### Practice

Answer:

1. What is the difference between health and readiness?
2. Why should request schemas limit input size?
3. What would you log for each prediction?
4. What should never be logged?

## Module 10 - Build Your Portfolio Pitch

Your portfolio pitch should connect your projects into one coherent story.

Template:

```text
My portfolio focuses on applied machine learning and data engineering for finance and operational decision-making. I built projects covering credit risk, fraud detection, forecasting, MLOps, NLP, NoSQL analytics, and lakehouse pipelines. Across the projects, I used real external datasets, built reproducible pipelines, served models with APIs, added tests, and documented limitations and governance.
```

### Practice

Write your own 30-second pitch.

Rules:

- No buzzword list
- No unsupported claims
- Mention real datasets
- Mention both modeling and engineering
- End with the kind of role you are targeting

## Module 11 - Align Projects to Job Descriptions

Different jobs care about different projects.

For data scientist roles, highlight:

- Modeling
- Evaluation metrics
- Feature engineering
- Business interpretation
- Experiments

For data engineer roles, highlight:

- Pipelines
- Data validation
- Storage layers
- Scheduling
- SQL/NoSQL design
- Scalability

For ML engineer roles, highlight:

- APIs
- Docker
- Tests
- Model artifacts
- CI/CD
- Monitoring

### Exercise

Find one job description.

Create a table:

```markdown
| Job Requirement | Matching Project Evidence |
| --- | --- |
| Build ML models | Credit Risk API, Fraud Detection API |
| Deploy models | FastAPI endpoints, Dockerfiles |
| Data pipelines | NYC Taxi Lakehouse, MLOps pipeline |
| NLP | Earnings Sentiment API |
```

Then choose which three projects you would mention first in an interview for that job.

## Module 12 - Create a Resume Project Section

A strong project section should be short and specific.

Example:

```markdown
DATA SCIENCE PROJECTS

Credit Risk API
- Built a loan default risk scoring API using real Home Credit application, bureau, and previous-loan data; engineered borrower-level risk features and served predictions through FastAPI.
- Added model metadata, validation checks, Docker packaging, and automated tests to simulate a production ML workflow.

NYC Taxi Lakehouse and Demand Forecasting
- Built a lakehouse-style ETL pipeline using official NYC taxi trip data and weather enrichment; created hourly demand marts and trained zone-level forecasting models.
- Exposed forecast and anomaly outputs through API endpoints with reproducible data processing and test coverage.

NLP Earnings Sentiment API
- Trained a financial sentiment classifier using Financial PhraseBank data with TF-IDF and finance lexicon features.
- Served sentence, transcript, and batch predictions through FastAPI with model card documentation and evaluation metadata.
```

### Practice

Write your own project section with:

- Three projects
- Two bullets per project
- One GitHub link per project

## Module 13 - Mock Interview Drill

Record yourself answering these questions:

1. Tell me about your best project.
2. Why did you choose that dataset?
3. What was the hardest technical part?
4. What metric did you optimize?
5. How did you test the project?
6. How would you deploy it?
7. What are the limitations?
8. What would you improve next?

Review the recording.

Score yourself:

```markdown
| Area | Score 1-5 | Notes |
| --- | ---: | --- |
| Clear problem statement |  |  |
| Technical depth |  |  |
| Business relevance |  |  |
| Honest limitations |  |  |
| Concise delivery |  |  |
```

### Exercise

Repeat the answer until you can explain the project in under two minutes without reading.

## Final Assignment

Create a complete interview packet.

Your packet must include:

- Project story inventory table
- Six resume bullets
- Three one-minute project pitches
- Two technical deep dives
- Four STAR stories
- One 30-second portfolio pitch
- Job-description alignment table
- Resume project section

## Rubric

| Area | Strong Submission |
| --- | --- |
| Accuracy | Every claim is supported by a repo artifact |
| Clarity | A nontechnical recruiter can understand the project value |
| Technical Depth | A hiring manager can see modeling, engineering, and evaluation choices |
| Metrics | Metrics are specific and not exaggerated |
| Storytelling | Answers follow clear structures and stay concise |
| Job Alignment | Projects are matched to role requirements |

## Interview Questions to Practice

Use these for weekly practice:

1. Which project are you most proud of?
2. What was the business problem?
3. What real dataset did you use?
4. How did you clean the data?
5. What features did you engineer?
6. Why did you choose the model?
7. What metric mattered most?
8. What surprised you during the project?
9. How did you test the system?
10. How would this change in production?
11. What would you monitor?
12. What are the limitations?
13. What would you improve next?
14. How does this project relate to the job?
15. What did you learn?

## Completion Checklist

You are done when:

- You can explain each top project in one minute
- You can deep-dive two projects technically
- Your resume bullets are specific and supported
- Your metrics are honest
- Your STAR stories are written
- Your portfolio pitch sounds natural
- Your project section matches your GitHub repos
- You can explain limitations without sounding defensive

