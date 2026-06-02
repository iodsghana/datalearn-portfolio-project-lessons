# Lesson 11 - Cloud API Deployment and Monitoring

In this lesson, you will take a completed FastAPI data project and prepare it for cloud-style deployment.

This lesson does not require you to keep a paid cloud server running. The goal is to learn the deployment workflow, document the architecture honestly, and understand what production monitoring would require.

You can use any API project from earlier lessons:

- Credit Risk API
- Fraud Detection API
- Student Loan Forecasting API
- NLP Earnings Sentiment API
- NYC Taxi Demand API

## Learning Goals

By the end of this lesson, you should be able to:

- Explain how a model API is deployed with Docker
- Run a FastAPI app in a production-like mode
- Configure environment variables
- Deploy a container to an EC2 instance
- Add health and readiness checks for deployment
- Understand basic cloud networking
- Create monitoring logs and metrics
- Document deployment evidence honestly
- Explain how you would scale or secure the API

## What You Are Building

You will create a deployment package for one existing API project.

Final artifacts:

- Production-style Dockerfile
- `.env.example`
- Deployment commands
- EC2 deployment notes
- Health check documentation
- Monitoring plan
- Rollback plan
- Deployment evidence section in README

## Module 1 - Choose the API Project

Pick one API project.

Recommended:

```text
credit-risk-api
```

or:

```text
NLP-Earnings-Sentiment
```

Create:

```text
deployment/
  ec2_deploy.md
  monitoring_plan.md
  rollback_plan.md
```

### Practice

Answer:

1. What endpoint makes predictions?
2. What endpoint checks health?
3. What model artifact must exist before the API is ready?
4. What port does the API use locally?

## Module 2 - Confirm Local Production Run

Before cloud deployment, the API must run locally.

Run:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest tests/ -q
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

Check:

```text
http://127.0.0.1:8000/healthz
http://127.0.0.1:8000/readyz
```

### Practice

Write down:

- Test result
- Health response
- Readiness response
- One prediction response

## Module 3 - Create a Production Dockerfile

Use this as a starting point:

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build:

```bash
docker build -t portfolio-api .
```

Run:

```bash
docker run --rm -p 8000:8000 portfolio-api
```

### Exercise

Test the container:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
```

If `curl` is unavailable on Windows, open the URLs in a browser.

## Module 4 - Add Runtime Configuration

Create `.env.example`:

```text
APP_ENV=production
APP_PORT=8000
LOG_LEVEL=INFO
MODEL_PATH=models/model.pkl
MODEL_METADATA_PATH=models/model_metadata.json
```

Add `.env` to `.gitignore`:

```text
.env
```

### Why This Matters

Configuration changes between local development and cloud deployment. The code should not require editing just because the environment changed.

### Practice

Answer:

1. Which settings are safe to commit?
2. Which settings should be private?
3. Why should cloud credentials never be placed in code?

## Module 5 - Understand Basic EC2 Deployment

EC2 is a virtual server. A simple deployment flow is:

```text
GitHub repo
  -> EC2 instance
  -> Docker build
  -> Docker run
  -> API available on port 8000
```

For a portfolio demo, use a small free-tier style instance when available. Stop the instance when you are done testing.

Create `deployment/ec2_deploy.md`:

```markdown
# EC2 Deployment Notes

## Architecture

GitHub repository -> EC2 Ubuntu server -> Docker container -> FastAPI app on port 8000

## Instance Setup

- OS: Ubuntu
- Inbound ports:
  - 22 for SSH
  - 8000 for API demo

## Commands

```bash
sudo apt update
sudo apt install -y docker.io git
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ubuntu
```

Reconnect after adding the Docker group.

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
docker build -t portfolio-api .
docker run -d --name portfolio-api -p 8000:8000 portfolio-api
```

## Verification

```bash
curl http://PUBLIC_IP:8000/healthz
curl http://PUBLIC_IP:8000/readyz
```

## Shutdown

```bash
docker stop portfolio-api
docker rm portfolio-api
```
```

### Practice

Replace:

- `YOUR_USERNAME`
- `YOUR_REPO`
- `PUBLIC_IP`

## Module 6 - Add a Safer Container Run Command

Production-like containers should restart if the server reboots.

Run:

```bash
docker run -d \
  --name portfolio-api \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env \
  portfolio-api
```

On PowerShell, use one line:

```powershell
docker run -d --name portfolio-api --restart unless-stopped -p 8000:8000 --env-file .env portfolio-api
```

### Exercise

Explain:

1. What does `-d` do?
2. What does `-p 8000:8000` do?
3. What does `--restart unless-stopped` do?
4. Why use `--env-file`?

## Module 7 - Add Logs

View container logs:

```bash
docker logs portfolio-api
```

Follow logs:

```bash
docker logs -f portfolio-api
```

In your API, log important events:

```python
logger.info("Model loaded")
logger.info("Prediction request completed")
logger.warning("Readiness check failed")
logger.error("Prediction failed")
```

### What to Monitor

For a model API, monitor:

- Request count
- Error count
- Latency
- Readiness failures
- Prediction distribution
- Input validation failures
- Model version

### Practice

Make one prediction request, then inspect Docker logs.

Write down what appeared.

## Module 8 - Create a Monitoring Plan

Create `deployment/monitoring_plan.md`:

```markdown
# Monitoring Plan

## Service Metrics

- Request count
- Error rate
- Response latency
- Health check status
- Readiness check status

## Model Metrics

- Prediction distribution
- Average confidence
- Drift in important input features
- Missing-value rate
- Model version served

## Alerts

- API returns 5xx errors repeatedly
- Readiness check fails
- Latency is unusually high
- Prediction distribution changes sharply
- Input schema validation failures increase

## Logs to Keep

- Timestamp
- Endpoint
- Response status
- Latency
- Model version

## Logs to Avoid

- Passwords
- API keys
- Full private customer records
- Sensitive financial identifiers
```

### Exercise

Add three project-specific monitoring signals.

For credit risk:

- Default probability distribution
- Missing bureau feature rate
- High-risk prediction percentage

For NLP:

- Average text length
- Sentiment distribution
- Low-confidence prediction percentage

## Module 9 - Create a Rollback Plan

Rollback means returning to a previous working version if deployment fails.

Create `deployment/rollback_plan.md`:

```markdown
# Rollback Plan

## Failure Cases

- Docker image fails to build
- Container starts but readiness check fails
- Prediction endpoint returns errors
- New model metadata is missing
- Latency becomes unacceptable

## Rollback Steps

1. Stop the current container.
2. Start the previous working image or commit.
3. Verify `/healthz`.
4. Verify `/readyz`.
5. Run one sample prediction.
6. Document the issue before retrying deployment.

## Commands

```bash
docker stop portfolio-api
docker rm portfolio-api
git checkout PREVIOUS_COMMIT
docker build -t portfolio-api .
docker run -d --name portfolio-api -p 8000:8000 portfolio-api
```
```

### Practice

Find the previous commit hash:

```bash
git log --oneline -5
```

Write which commit you would roll back to.

## Module 10 - Add Security Notes

A portfolio API is not automatically secure.

Add this to the README:

```markdown
## Security Notes

This API is structured for demonstration and portfolio review. A production deployment should add:

- Authentication
- HTTPS
- Rate limiting
- Request logging controls
- Secret management
- Network restrictions
- Monitoring and alerting
```

### Practice

Answer:

1. Why is an open prediction API risky?
2. Why should HTTPS be used?
3. What should rate limiting protect against?
4. What input data should never be logged?

## Module 11 - Document Deployment Evidence

Add a README section:

```markdown
## Deployment Evidence

The API is containerized with Docker and can be deployed to an Ubuntu EC2 instance.

Verified checks:

- `docker build` completes
- `/healthz` returns `{"status": "ok"}`
- `/readyz` confirms the model artifact is available
- Sample prediction request returns a valid response

See:

- `deployment/ec2_deploy.md`
- `deployment/monitoring_plan.md`
- `deployment/rollback_plan.md`
```

If you actually deploy it temporarily, add:

```markdown
Tested on AWS EC2 for demonstration on DATE. The instance was stopped after validation to avoid ongoing cost.
```

### Exercise

Add deployment evidence to one project README.

## Module 12 - Compare Deployment Options

You should understand the difference between common deployment choices.

| Option | Good For | Tradeoff |
| --- | --- | --- |
| EC2 + Docker | Learning fundamentals and simple demos | You manage the server |
| ECS/Fargate | Managed container deployment | More setup |
| Lambda + API Gateway | Lightweight serverless APIs | Cold starts and package limits |
| SageMaker Endpoint | ML model hosting and monitoring | More expensive and specialized |
| Render/Railway/Fly.io | Simple demos | Less enterprise cloud signal |

### Practice

Choose the best option for:

1. A resume demo
2. A production financial model
3. A low-cost student project
4. A high-traffic model API

Explain each choice.

## Final Assignment

Deploy or deployment-package one API project.

Required:

- Dockerfile works locally
- `.env.example` exists
- `/healthz` endpoint works
- `/readyz` endpoint works
- `deployment/ec2_deploy.md`
- `deployment/monitoring_plan.md`
- `deployment/rollback_plan.md`
- README deployment evidence section
- Security notes

Optional:

- Temporarily deploy to EC2
- Add screenshots
- Add GitHub Actions Docker build
- Add request logging middleware
- Add simple rate limiting

## Rubric

| Area | Strong Submission |
| --- | --- |
| Docker | Image builds and container runs |
| Readiness | API can verify model or pipeline availability |
| Deployment Docs | EC2 commands are clear and reproducible |
| Monitoring | Service and model monitoring signals are listed |
| Rollback | Failure cases and recovery commands are documented |
| Security | Limitations and production security needs are honest |
| Evidence | README shows what was verified |

## Interview Questions

Practice answering:

1. How would you deploy this API?
2. Why use Docker?
3. What is the difference between EC2 and ECS?
4. What is the difference between health and readiness?
5. What would you monitor after deployment?
6. How would you roll back a bad model?
7. How would you secure the API?
8. What logs would you keep?
9. What logs would you avoid?
10. How would you reduce cloud costs for a portfolio demo?

## Completion Checklist

You are done when:

- Docker image builds
- Container runs locally
- Health endpoint works
- Readiness endpoint works
- EC2 deployment notes exist
- Monitoring plan exists
- Rollback plan exists
- README includes deployment evidence
- README includes security notes
- Claims are honest about whether the API is actually deployed

