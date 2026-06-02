# Lesson 08 - GitHub Portfolio Packaging for Data Projects

In this lesson, you will learn how to package data science, data engineering, analytics, NLP, and MLOps projects so recruiters and hiring managers can understand your work quickly.

This lesson is different from the earlier lessons. You are not training a new model. You are turning finished technical projects into a professional portfolio.

That matters because many strong projects fail to impress because the GitHub presentation is weak.

By the end, you will build:

- A clean GitHub profile README
- Strong project README files
- Recruiter-friendly project summaries
- Metrics tables
- Project architecture sections
- Demo screenshots or API examples
- A simple portfolio website
- Resume bullets based on real project work

## Learning Goals

By the end of this lesson, you should be able to explain:

- What recruiters look for in a technical portfolio
- Why project presentation matters as much as code quality
- How to write a strong project README
- How to show business impact without exaggerating
- How to organize pinned GitHub repositories
- How to create a simple GitHub Pages portfolio site
- How to translate project work into resume bullets
- How to avoid common beginner portfolio mistakes

## Who This Lesson Is For

This lesson is for students who have already built at least two technical projects.

Recommended minimum:

- One machine learning project
- One API, data engineering, analytics, or dashboard project

If you have completed earlier lessons in this repo, you can use those projects.

## What Recruiters Notice First

Recruiters and hiring managers often scan quickly. They may not read every file.

They usually look for:

- Clear project title
- Business problem
- Real dataset source
- Technical approach
- Measurable results
- How to run the project
- Evidence of testing or deployment
- Clean repository structure
- Screenshots, API docs, or demo examples

Your goal is to make those signals obvious.

## Module 1 - Audit Your Existing Projects

Create a document called `portfolio_audit.md`.

Add this table:

```markdown
| Project | Real Data Source | Model/API/Pipeline | Metrics | Tests | README Quality | Needs Work |
| --- | --- | --- | --- | --- | --- | --- |
| Credit Risk API | Home Credit | FastAPI model API | AUC, recall | Yes | Strong | Add screenshot |
| Fraud Detection API | Home Credit | Risk screening API | AUC, recall | Yes | Strong | Add threshold chart |
| Student Loan Forecasting | Public loan/macro data | Forecast API | MAE/RMSE | Yes | Strong | Add scenario graphic |
```

Fill in your own projects.

### Practice

Answer:

1. Which project is your strongest technical project?
2. Which project is easiest for a recruiter to understand?
3. Which project has the weakest README?
4. Which project has the best business story?

## Module 2 - Choose Your Pinned Repositories

Do not pin every project. Pin the projects that tell the strongest story.

A strong data portfolio might pin:

- Credit risk API
- Fraud detection system
- MLOps project
- Data lakehouse or data engineering project
- NLP project
- Database or analytics project

### Selection Rule

Each pinned project should show a different strength.

Example:

| Project | Skill Signal |
| --- | --- |
| Credit Risk API | Supervised ML, finance, API deployment |
| NYC Taxi Lakehouse | Data engineering, ETL, forecasting |
| MLOps SageMaker Churn and CLV | Model governance, deployment, monitoring |
| NLP Earnings Sentiment | NLP, text classification, FastAPI |
| NoSQL Retail Analytics | Database design, MongoDB, analytics APIs |
| Student Loan Forecasting | Time series, risk, scenario modeling |

### Exercise

Pick six projects to pin.

For each one, write one sentence explaining why it deserves to be pinned.

## Module 3 - Write a Strong Project README

Every major project should have a README that answers these questions:

- What problem does this solve?
- What real dataset did you use?
- What did you build?
- What are the results?
- How can someone run it?
- What files should they inspect?
- What are the limitations?

Use this structure:

```markdown
# Project Title

One-sentence summary of what the project does.

## Business Problem

Explain the real-world problem in plain English.

## Data

Describe the real dataset and include the source link.

## Approach

Explain the modeling, analytics, or engineering workflow.

## Results

| Metric | Value |
| --- | ---: |
| Test AUC | 0.86 |
| Recall at threshold | 0.74 |
| API latency | 32 ms |

## How to Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
pytest tests/ -q
uvicorn api.app:app --reload
```

## Project Structure

```text
api/
data/
models/
src/
tests/
README.md
requirements.txt
```

## Governance and Limitations

Explain what the model should and should not be used for.
```

### Practice

Choose one project and rewrite the first three sections:

- One-sentence summary
- Business problem
- Data

Keep the language simple enough for a recruiter and strong enough for an engineering manager.

## Module 4 - Write Better Project Summaries

Weak summary:

```text
This project predicts credit risk using machine learning.
```

Stronger summary:

```text
Built a production-style credit risk API using real Home Credit application, bureau, and previous-loan data. The project includes feature engineering, model validation, threshold analysis, FastAPI scoring, tests, Docker packaging, and governance documentation.
```

Why it is stronger:

- It names the dataset
- It says production-style
- It mentions feature engineering
- It mentions API serving
- It mentions tests and governance

### Formula

Use this formula:

```text
Built a [type of system] using [real data source] to solve [business problem]. The project includes [3-5 technical signals] and produces [metric/output/business result].
```

### Exercise

Write summaries for three projects:

1. One ML API project
2. One data engineering project
3. One analytics or NLP project

## Module 5 - Add Metrics Without Exaggerating

Good metrics:

- Test AUC
- Macro F1
- Recall at selected threshold
- MAE or RMSE
- API latency
- Number of rows processed
- Number of features engineered
- Number of tests

Avoid unsupported claims:

- "Improved revenue by 30%" if you did not measure revenue
- "FAANG-level" inside the README
- "Production deployed" if it only runs locally
- "Real-time system" if there is no API or streaming component

Better wording:

```text
Simulated a production scoring workflow with FastAPI and Docker.
```

```text
Built a deployment-ready API structure with health checks, model metadata, and tests.
```

```text
Processed 300K+ loan applications from the Home Credit dataset.
```

### Practice

For each project, list:

- One model metric
- One engineering metric
- One business interpretation

Example:

```text
Model metric: Test ROC-AUC = 0.86
Engineering metric: API returns predictions in under 100 ms locally
Business interpretation: The model ranks borrowers by relative default risk for review workflows
```

## Module 6 - Show Architecture Clearly

A simple architecture section helps readers understand the workflow.

Example:

```text
Raw data
  -> validation
  -> feature engineering
  -> model training
  -> model artifact
  -> FastAPI service
  -> prediction response
```

You can include this in a README:

```markdown
## Architecture

```text
data/
  raw source files
src/
  download_data.py
  train.py
  predict.py
api/
  app.py
models/
  trained model artifact
monitoring/
  model card and dashboard
tests/
  automated checks
```
```

### Exercise

Draw the architecture for one of your projects using plain text.

Include:

- Data source
- Processing step
- Model or analytics step
- Output
- API, dashboard, or report

## Module 7 - Add Screenshots and Demo Evidence

Screenshots make a project easier to trust.

Good screenshots:

- FastAPI Swagger docs
- Model dashboard
- Confusion matrix
- Forecast plot
- Data quality report
- Terminal test result
- Example API response
- Database query output

Create an `assets/` folder in each project:

```bash
mkdir assets
```

Suggested files:

```text
assets/api_docs.png
assets/model_results.png
assets/test_results.png
assets/architecture.png
```

Add screenshots to the README:

```markdown
## Demo

![API docs](assets/api_docs.png)

![Model dashboard](assets/model_results.png)
```

### Practice

For one project, capture at least two screenshots:

- One showing that the project runs
- One showing model or analytics results

## Module 8 - Create a GitHub Profile README

A GitHub profile README appears on your GitHub homepage when you create a repository with the same name as your username.

For example, if your username is:

```text
myusername
```

Create a repo named:

```text
myusername
```

Then add a `README.md`.

Use this structure:

```markdown
# Your Name

Data Scientist focused on machine learning, financial analytics, data engineering, and production ML systems.

## Featured Projects

| Project | What It Shows |
| --- | --- |
| Credit Risk API | Supervised ML, finance, FastAPI, model governance |
| NYC Taxi Lakehouse | ETL, lakehouse design, forecasting, anomaly detection |
| NLP Earnings Sentiment | Financial NLP, text classification, model serving |

## Technical Skills

- Python, SQL, Pandas, NumPy
- Scikit-learn, XGBoost, forecasting
- FastAPI, Docker, GitHub Actions
- AWS, SageMaker, S3
- MongoDB, PostgreSQL

## Contact

- LinkedIn: your link
- GitHub: your link
- Email: your email
```

### Practice

Write your GitHub profile README in one page.

Avoid:

- Long autobiography
- Too many badges
- Unverified claims
- Projects with no links

## Module 9 - Build a Simple Portfolio Website

Create a folder:

```bash
mkdir portfolio-website
cd portfolio-website
```

Create:

```text
index.html
style.css
script.js
assets/
```

Create `index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Your Name | Data Scientist</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header class="site-header">
    <h1>Your Name</h1>
    <p>Data Scientist | Machine Learning | Data Engineering | Analytics</p>
  </header>

  <nav class="nav">
    <a href="#about">About</a>
    <a href="#projects">Projects</a>
    <a href="#skills">Skills</a>
    <a href="#contact">Contact</a>
  </nav>

  <main>
    <section id="about">
      <h2>About</h2>
      <p>
        I build machine learning, analytics, and data engineering projects that connect real data,
        practical modeling, and deployment-ready software.
      </p>
    </section>

    <section id="projects">
      <h2>Projects</h2>

      <article class="project">
        <h3>Credit Risk API</h3>
        <p>
          Production-style loan default model using real credit application data,
          feature engineering, model validation, FastAPI scoring, and tests.
        </p>
        <a href="https://github.com/YOUR_USERNAME/credit-risk-api">View Code</a>
      </article>

      <article class="project">
        <h3>NYC Taxi Lakehouse and Demand Forecasting</h3>
        <p>
          Data lakehouse project using official taxi data, ETL marts, demand forecasting,
          anomaly detection, and API endpoints.
        </p>
        <a href="https://github.com/YOUR_USERNAME/NYC-Taxi-Lakehouse-and-Demand-Forecasting">View Code</a>
      </article>

      <article class="project">
        <h3>NLP Earnings Sentiment API</h3>
        <p>
          Financial text classifier trained on real Financial PhraseBank data with model metadata,
          governance documentation, and FastAPI serving.
        </p>
        <a href="https://github.com/YOUR_USERNAME/NLP-Earnings-Sentiment">View Code</a>
      </article>
    </section>

    <section id="skills">
      <h2>Skills</h2>
      <ul>
        <li>Python, SQL, Pandas, NumPy</li>
        <li>Scikit-learn, XGBoost, forecasting, NLP</li>
        <li>FastAPI, Docker, Git, GitHub Actions</li>
        <li>AWS, SageMaker, S3, MongoDB</li>
      </ul>
    </section>

    <section id="contact">
      <h2>Contact</h2>
      <p>
        <a href="https://github.com/YOUR_USERNAME">GitHub</a>
        |
        <a href="https://linkedin.com/in/YOUR_PROFILE">LinkedIn</a>
      </p>
    </section>
  </main>

  <footer>
    <p>(c) 2026 Your Name</p>
  </footer>

  <script src="script.js"></script>
</body>
</html>
```

Create `style.css`:

```css
body {
  margin: 0;
  font-family: Arial, sans-serif;
  color: #1f2937;
  background: #f5f7fb;
}

.site-header {
  padding: 36px 24px;
  color: white;
  background: #1f2937;
  text-align: center;
}

.site-header h1 {
  margin: 0 0 8px;
  font-size: 36px;
}

.site-header p {
  margin: 0;
}

.nav {
  display: flex;
  justify-content: center;
  gap: 20px;
  padding: 12px;
  background: #111827;
}

.nav a {
  color: white;
  text-decoration: none;
  font-weight: 700;
}

main {
  max-width: 960px;
  margin: 0 auto;
  padding: 32px 20px;
}

section {
  margin-bottom: 36px;
}

.project {
  margin-bottom: 16px;
  padding: 18px;
  background: white;
  border: 1px solid #dbe3ef;
  border-radius: 8px;
}

.project h3 {
  margin-top: 0;
}

.project a {
  color: #2563eb;
  font-weight: 700;
  text-decoration: none;
}

footer {
  padding: 20px;
  color: white;
  background: #1f2937;
  text-align: center;
}

@media (max-width: 640px) {
  .nav {
    flex-wrap: wrap;
  }
}
```

Create `script.js`:

```javascript
document.querySelectorAll(".nav a").forEach((anchor) => {
  anchor.addEventListener("click", (event) => {
    event.preventDefault();
    document.querySelector(anchor.getAttribute("href")).scrollIntoView({
      behavior: "smooth",
    });
  });
});
```

Open `index.html` in your browser.

### Practice

Replace every placeholder:

- `Your Name`
- `YOUR_USERNAME`
- `YOUR_PROFILE`
- Project names
- Project links
- Skill list

## Module 10 - Deploy the Portfolio with GitHub Pages

Create a new GitHub repository called:

```text
portfolio-website
```

Push your files:

```bash
git init
git add .
git commit -m "Create portfolio website"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/portfolio-website.git
git push -u origin main
```

Then enable GitHub Pages from the repository settings.

Your site will usually look like:

```text
https://YOUR_USERNAME.github.io/portfolio-website/
```

### Practice

After deployment:

1. Open the live URL
2. Click every project link
3. Check the site on mobile
4. Fix any broken links

## Module 11 - Write Resume Bullets from Projects

A project bullet should include:

- Action
- Technical method
- Data or scale
- Result or output

Weak:

```text
Built a machine learning model for credit risk.
```

Strong:

```text
Built a credit risk scoring API using real Home Credit application, bureau, and prior-loan data; engineered borrower-level risk features and served predictions through FastAPI with model metadata and automated tests.
```

Another example:

```text
Developed an NYC taxi demand lakehouse using official trip data, hourly zone-demand marts, weather enrichment, forecasting models, anomaly detection, and API endpoints for operational demand planning.
```

### Practice

Write two bullets for each of your top three projects:

- One technical bullet
- One business-impact bullet

## Module 12 - Create a Recruiter Review Checklist

Before you share your portfolio, check every repo.

Use this checklist:

```markdown
## Portfolio Review Checklist

- [ ] README has a clear one-sentence summary
- [ ] Business problem is easy to understand
- [ ] Real data source is named and linked
- [ ] Metrics are included
- [ ] How-to-run commands are included
- [ ] Project structure is shown
- [ ] Tests are present or limitations are explained
- [ ] Screenshots or demo outputs are included
- [ ] No private data is exposed
- [ ] No fake claims are included
- [ ] Repo name is professional
- [ ] Latest commit message is clean
```

### Exercise

Review one of your repositories using this checklist.

Mark each item:

- Pass
- Needs work
- Not applicable

Then fix at least three `Needs work` items.

## Module 13 - Common Mistakes to Avoid

Avoid these:

- README only says "run notebook"
- No explanation of the business problem
- No data source
- No metrics
- No tests
- Too many half-finished repos pinned
- Project names are vague
- Code depends on private local paths
- Resume claims do not match the repo
- Screenshots show broken apps

Better choices:

- Pin fewer, stronger repos
- Use real external datasets
- Show deployment-ready structure
- Include tests
- Add model cards for ML projects
- Explain limitations clearly

### Practice

Find one mistake in your own portfolio and fix it.

## Module 14 - Final Portfolio Assignment

Package three projects completely.

For each project, include:

- Strong README
- Real data source
- Metrics table
- How-to-run section
- Architecture section
- Tests or validation checks
- Screenshot or API example
- Governance or limitations section

Then create:

- GitHub profile README
- Portfolio website
- Three resume bullets

## Final Deliverables

Submit:

- Link to GitHub profile
- Links to six pinned repositories
- Link to portfolio website
- Three strongest project README files
- Six resume bullets based on project work
- Screenshot of at least one API docs page or dashboard

## Rubric

| Area | Strong Submission |
| --- | --- |
| Project Selection | Pinned repos show different strengths |
| README Quality | Clear business problem, data, approach, metrics, and run instructions |
| Technical Evidence | Tests, APIs, dashboards, model cards, or deployment structure are visible |
| Honesty | Claims are specific and supported by project artifacts |
| Recruiter Usability | A recruiter can understand the portfolio in under two minutes |
| Website | Portfolio site links to real repos and is easy to scan |
| Resume Alignment | Resume bullets match actual GitHub work |

## Interview Questions

Practice answering:

1. Which project best represents your technical ability?
2. Which project best shows business thinking?
3. Why did you choose these projects for your GitHub pins?
4. How do your project metrics support your claims?
5. What would you improve in your portfolio next?
6. How do your GitHub projects connect to the jobs you want?
7. Which project was hardest to build, and why?
8. How do you know your models are not just notebook experiments?
9. What evidence shows that your code is testable?
10. How would you explain your portfolio to a nontechnical recruiter?

## Completion Checklist

You are done when:

- Six pinned repos are selected intentionally
- Your GitHub profile README is complete
- At least three project READMEs are polished
- Each top project has data, metrics, run instructions, and limitations
- Your portfolio website is created
- Project links work
- Resume bullets match actual project work
- No private data or secrets are committed
- Your portfolio can be scanned quickly by a recruiter
