# Lesson 07 - NLP Earnings Sentiment API

In this lesson, you will build a production-style NLP project that classifies financial sentences and earnings-call text as negative, neutral, or positive.

This is not a notebook-only project. You will build the full workflow:

- Download a real external financial NLP dataset
- Prepare labels and metadata
- Engineer finance-specific text features
- Train and evaluate a multiclass model
- Save model artifacts and metadata
- Create a model card
- Build a FastAPI inference service
- Score individual sentences, batches, and earnings-call transcripts
- Add tests for data, features, predictions, and API behavior
- Package the project for a portfolio

Finished project reference:

https://github.com/iodsghana/NLP-Earnings-Sentiment

## Real Dataset

This project uses Financial PhraseBank, a real external dataset of finance-domain sentences labeled by human annotators as negative, neutral, or positive.

Dataset page:

https://huggingface.co/datasets/financial_phrasebank

The finished project downloads the dataset from:

https://huggingface.co/datasets/takala/financial_phrasebank

You will use the `sentences_allagree` configuration because it keeps examples where annotators agreed strongly. That gives you fewer rows than the larger configurations, but the labels are cleaner.

## What You Are Building

Imagine your company reviews earnings-call transcripts from public companies. Analysts want a quick way to triage the tone of the call:

- Is management language mostly positive?
- Are there warning signs in the transcript?
- Which sentences sound most positive or negative?
- Can we expose the model through an API for other tools?

By the end, your project will support endpoints like:

- `GET /healthz`
- `GET /readyz`
- `GET /v1/model/info`
- `POST /v1/sentence`
- `POST /v1/call`
- `POST /v1/batch`

## Learning Goals

By the end of this lesson, you should be able to explain:

- Why finance NLP needs domain-specific evaluation
- How labeled text data becomes model-ready data
- What TF-IDF features capture
- Why finance lexicon features can help
- How to evaluate a multiclass classifier
- What a model card communicates
- How to serve a trained NLP model with FastAPI
- How to test an NLP project beyond just checking accuracy

## Project Structure

Create this structure:

```text
nlp-earnings-sentiment/
  api/
    app.py
  data/
    README.md
  models/
  monitoring/
  src/
    download_data.py
    predict.py
    train.py
  tests/
    test_sentiment.py
  .gitignore
  Dockerfile
  README.md
  requirements.txt
```

## Module 1 - Create the Environment

Create your project folder:

```bash
mkdir nlp-earnings-sentiment
cd nlp-earnings-sentiment
```

Create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Create `requirements.txt`:

```text
fastapi
uvicorn
pandas
numpy
scikit-learn
scipy
joblib
matplotlib
pytest
httpx
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your folders:

```bash
mkdir api data models monitoring src tests
```

Create `.gitignore`:

```text
.venv/
__pycache__/
.pytest_cache/
*.pyc
```

### Practice

Answer these before moving on:

1. Why should `.venv/` not be committed to GitHub?
2. Why should model files be saved in a predictable folder?
3. What is the difference between `src/` and `api/` in this project?

## Module 2 - Download the Real Dataset

Create `src/download_data.py`:

```python
"""Download real external financial NLP data."""

from __future__ import annotations

import io
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

ZIP_URL = (
    "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/"
    "data/FinancialPhraseBank-v1.0.zip"
)

CONFIG_FILES = {
    "sentences_allagree": "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt",
    "sentences_75agree": "FinancialPhraseBank-v1.0/Sentences_75Agree.txt",
    "sentences_66agree": "FinancialPhraseBank-v1.0/Sentences_66Agree.txt",
    "sentences_50agree": "FinancialPhraseBank-v1.0/Sentences_50Agree.txt",
}


def download_financial_phrasebank(config: str = "sentences_allagree") -> Path:
    """Download Financial PhraseBank and save it as a CSV."""
    if config not in CONFIG_FILES:
        raise ValueError(f"Unsupported config: {config}")

    with urlopen(ZIP_URL, timeout=30) as response:
        archive_bytes = response.read()

    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        raw = archive.read(CONFIG_FILES[config]).decode("ISO-8859-1")

    rows = []
    for line in raw.splitlines():
        if not line.strip() or "@" not in line:
            continue
        text, label_name = line.rsplit("@", 1)
        label_name = label_name.strip().lower()
        if label_name not in INV_LABEL_MAP:
            continue
        rows.append({"text": text.strip(), "label": INV_LABEL_MAP[label_name]})

    frame = pd.DataFrame(rows)
    frame["label_name"] = frame["label"].map(LABEL_MAP)
    frame["source"] = "Financial PhraseBank"
    frame["config"] = config

    out = DATA_DIR / "financial_phrasebank.csv"
    frame.to_csv(out, index=False)
    return out


def main() -> None:
    path = download_financial_phrasebank()
    print(f"Downloaded real dataset -> {path}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
python src/download_data.py
```

Expected result:

```text
data/financial_phrasebank.csv
```

### Why This Matters

Resume projects are stronger when data ingestion is reproducible. Instead of saying, "I downloaded a file manually," your project shows exactly how the dataset enters the pipeline.

### Practice

Open the CSV and answer:

1. How many rows are in the dataset?
2. What are the three labels?
3. Why might `sentences_allagree` be better for a portfolio project than noisier labels?

## Module 3 - Understand the Labels

The label mapping is:

```python
LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}
```

This mapping matters because machine learning models usually train on numeric targets, but API users need human-readable outputs.

Example:

```text
"Operating profit increased after strong demand."
```

This should likely be positive.

Example:

```text
"The company warned that weak demand may pressure margins."
```

This should likely be negative or neutral depending on wording.

### Exercise

Write five finance sentences:

- Two positive
- Two negative
- One neutral

Do not use obvious toy examples only. Make them sound like something from an earnings release.

## Module 4 - Create Finance Lexicon Features

TF-IDF captures word patterns, but finance has special language. Words like `growth`, `profit`, `loss`, `uncertainty`, and `impairment` carry useful signal.

In `src/train.py`, start with imports and lexicon lists:

```python
"""Train a financial text sentiment model."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MONITORING_DIR = ROOT / "monitoring"

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

LM_POSITIVE = [
    "achieve", "achieved", "benefit", "better", "excellent", "favorable",
    "gain", "gains", "good", "growth", "improved", "improves", "positive",
    "profit", "profitable", "record", "strong", "success", "successful",
    "upgraded",
]

LM_NEGATIVE = [
    "adverse", "bankruptcy", "challenge", "challenging", "decline",
    "declined", "decrease", "decreased", "deterioration", "difficult",
    "loss", "losses", "negative", "risk", "risks", "weak", "weakness",
    "warning", "impairment", "uncertainty",
]

LM_UNCERTAINTY = [
    "approximately", "contingent", "could", "estimate", "may", "might",
    "possible", "risk", "uncertain", "uncertainty", "unknown", "volatile",
]
```

Add the feature function:

```python
def lm_features(texts: pd.Series | list[str]) -> np.ndarray:
    rows = []
    for text in pd.Series(texts).fillna(""):
        words = str(text).lower().replace(".", " ").replace(",", " ").split()
        total = max(len(words), 1)

        pos = sum(word in LM_POSITIVE for word in words)
        neg = sum(word in LM_NEGATIVE for word in words)
        unc = sum(word in LM_UNCERTAINTY for word in words)

        rows.append([
            pos,
            neg,
            unc,
            pos / total,
            neg / total,
            unc / total,
            (pos - neg) / max(pos + neg, 1),
        ])

    return np.asarray(rows, dtype=float)
```

### Concept Check

The final feature `(pos - neg) / max(pos + neg, 1)` is a simple polarity score.

If a sentence has:

- 3 positive words
- 1 negative word

Then:

```text
(3 - 1) / (3 + 1) = 0.5
```

That is positive.

### Practice

Use `lm_features` on these three sentences:

```python
pd.Series([
    "strong profit growth improved results",
    "losses declined because demand was weak",
    "",
])
```

Expected:

- Shape should be `(3, 7)`
- First row should have positive polarity
- Second row should have negative polarity
- Empty text should not create NaN values

## Module 5 - Load and Validate Data

Add this to `src/train.py`:

```python
def load_data() -> pd.DataFrame:
    path = DATA_DIR / "financial_phrasebank.csv"

    if not path.exists():
        from download_data import download_financial_phrasebank
        download_financial_phrasebank()

    df = pd.read_csv(path)
    required = {"text", "label", "label_name"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    return df.dropna(subset=["text", "label"]).reset_index(drop=True)
```

### Why Validate?

If your training code silently accepts bad data, your model can become unreliable without warning. A professional project checks that required columns exist before training.

### Exercise

Add one more validation rule:

```python
assert set(df["label"].unique()).issubset({0, 1, 2})
```

Then explain why this protects your training pipeline.

## Module 6 - Build TF-IDF Plus Lexicon Features

Add:

```python
def make_features(vectorizer: TfidfVectorizer, texts, fit: bool = False):
    text_matrix = vectorizer.fit_transform(texts) if fit else vectorizer.transform(texts)
    lexicon_matrix = csr_matrix(lm_features(texts))
    return hstack([text_matrix, lexicon_matrix])
```

### Explanation

This creates two types of features:

- TF-IDF word and phrase features
- Finance lexicon count and ratio features

The `hstack` function combines them side by side into one matrix.

### Practice

Answer:

1. Why do we call `fit_transform` only on training data?
2. Why do we call `transform` on validation and test data?
3. What kind of data leakage could happen if we fit on the full dataset?

## Module 7 - Train the Sentiment Model

Add the training function:

```python
def train() -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        stratify=train_df["label"],
        random_state=42,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=12000,
        strip_accents="unicode",
        sublinear_tf=True,
    )

    X_train = make_features(vectorizer, train_df["text"], fit=True)
    X_val = make_features(vectorizer, val_df["text"])
    X_test = make_features(vectorizer, test_df["text"])

    model = LogisticRegression(
        C=2.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, train_df["label"])

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_acc = accuracy_score(val_df["label"], val_pred)
    test_acc = accuracy_score(test_df["label"], test_pred)
    macro_f1 = f1_score(test_df["label"], test_pred, average="macro")

    report = classification_report(
        test_df["label"],
        test_pred,
        target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
        output_dict=True,
    )

    artifact = {
        "vectorizer": vectorizer,
        "model": model,
        "label_map": LABEL_MAP,
        "lm_positive": LM_POSITIVE,
        "lm_negative": LM_NEGATIVE,
        "lm_uncertainty": LM_UNCERTAINTY,
        "trained_at": datetime.now(UTC).isoformat(),
    }

    model_path = MODEL_DIR / "sentiment_model.pkl"
    joblib.dump(artifact, model_path)
    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()

    metadata = {
        "trained_at": artifact["trained_at"],
        "sha256": sha,
        "dataset": "Financial PhraseBank",
        "dataset_config": "sentences_allagree",
        "source_url": "https://huggingface.co/datasets/takala/financial_phrasebank",
        "n_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "metrics": {
            "validation_accuracy": round(float(val_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "test_macro_f1": round(float(macro_f1), 4),
        },
        "classification_report": report,
    }

    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    write_model_card(metadata)
    plot_dashboard(test_df["label"], test_pred, metadata)
    return metadata
```

Add the script entrypoint:

```python
if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
```

Do not run it yet. You still need the model card and dashboard functions.

### Why Logistic Regression?

Logistic regression is a strong baseline for TF-IDF text classification. It is fast, interpretable, and easy to deploy.

For a resume project, a clean classical NLP model with full production packaging is often stronger than a messy transformer notebook with no tests or API.

### Practice

Answer:

1. What does `class_weight="balanced"` do?
2. Why is macro F1 useful for multiclass sentiment?
3. Why should the test set be used only for final evaluation?

## Module 8 - Create a Model Card

Add this function:

```python
def write_model_card(metadata: dict) -> None:
    content = f"""# Model Card: Earnings Sentiment NLP

## Intended Use

Classify financial sentences and earnings-call excerpts as negative, neutral, or positive for analyst workflow support and transcript triage.

## Data

- Source: Financial PhraseBank, a real external dataset of finance-domain sentences annotated by humans.
- Configuration: `{metadata['dataset_config']}`.
- Rows: {metadata['n_rows']}.
- Source URL: {metadata['source_url']}.

## Evaluation

| Metric | Value |
| --- | ---: |
| Validation accuracy | {metadata['metrics']['validation_accuracy']} |
| Test accuracy | {metadata['metrics']['test_accuracy']} |
| Test macro F1 | {metadata['metrics']['test_macro_f1']} |

## Governance Notes

- This model is a text classification assistant, not financial advice.
- Outputs should not be used alone for trading or investment decisions.
- The model is trained on sentence-level financial news text; full earnings-call scoring aggregates sentence predictions heuristically.
- Production use should add fresh transcript validation, drift monitoring, and human review.
"""

    (MONITORING_DIR / "model_card.md").write_text(content, encoding="utf-8")
```

### Why This Matters

A model card explains what the model is for, what data it used, how it performed, and where it should not be used. This is a professional signal because real ML systems need governance, not just metrics.

### Exercise

Add a section called `Known Limitations` to the model card.

Include at least three limitations:

- The model is trained on sentence-level data
- It may not understand sarcasm or complex management language
- It should not be used as financial advice

## Module 9 - Create an Evaluation Dashboard

Add:

```python
def plot_dashboard(y_true, y_pred, metadata: dict) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks([0, 1, 2], [LABEL_MAP[i] for i in [0, 1, 2]], rotation=30)
    axes[0].set_yticks([0, 1, 2], [LABEL_MAP[i] for i in [0, 1, 2]])
    axes[0].set_title("Confusion Matrix")

    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, cm[i, j], ha="center", va="center")

    metrics = metadata["metrics"]
    axes[1].bar(
        metrics.keys(),
        metrics.values(),
        color=["#2563eb", "#16a34a", "#b45309"],
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Evaluation Metrics")
    axes[1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(MONITORING_DIR / "results_dashboard.png", dpi=150)
    plt.close(fig)
```

Now train:

```bash
python src/train.py
```

Expected outputs:

```text
models/sentiment_model.pkl
models/model_metadata.json
monitoring/model_card.md
monitoring/results_dashboard.png
```

### Practice

Open `models/model_metadata.json` and answer:

1. What is the test accuracy?
2. What is the macro F1 score?
3. Which label seems hardest to predict?
4. Why might neutral sentences be common in finance text?

## Module 10 - Build Prediction Utilities

Create `src/predict.py`:

```python
"""Inference utilities for the earnings sentiment model."""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "sentiment_model.pkl"
META_PATH = ROOT / "models" / "model_metadata.json"
```

Add a simple model registry:

```python
class SentimentRegistry:
    _artifact = None
    _metadata = None

    @classmethod
    def load(cls):
        if cls._artifact is None:
            cls._artifact = joblib.load(MODEL_PATH)
        if cls._metadata is None:
            cls._metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
        return cls._artifact, cls._metadata

    @classmethod
    def reload(cls):
        cls._artifact = None
        cls._metadata = None
        return cls.load()
```

### Why a Registry?

The registry prevents the API from loading the model from disk on every request. The first request loads the model, then later requests reuse it.

Add feature reconstruction:

```python
def _lm_features(texts: list[str], artifact: dict):
    pos_words = set(artifact["lm_positive"])
    neg_words = set(artifact["lm_negative"])
    unc_words = set(artifact["lm_uncertainty"])

    rows = []
    for text in texts:
        words = str(text).lower().replace(".", " ").replace(",", " ").split()
        total = max(len(words), 1)

        pos = sum(word in pos_words for word in words)
        neg = sum(word in neg_words for word in words)
        unc = sum(word in unc_words for word in words)

        rows.append([
            pos,
            neg,
            unc,
            pos / total,
            neg / total,
            unc / total,
            (pos - neg) / max(pos + neg, 1),
        ])

    return np.asarray(rows, dtype=float)


def _matrix(texts: list[str], artifact: dict):
    return hstack([
        artifact["vectorizer"].transform(texts),
        csr_matrix(_lm_features(texts, artifact)),
    ])
```

Add sentence prediction:

```python
def predict_sentence(text: str) -> dict:
    artifact, metadata = SentimentRegistry.load()
    X = _matrix([text], artifact)

    probs = artifact["model"].predict_proba(X)[0]
    classes = artifact["model"].classes_.tolist()
    label_map = {int(k): v for k, v in artifact["label_map"].items()}

    pred_idx = int(np.argmax(probs))
    pred_cls = int(classes[pred_idx])

    def prob_for(label: int) -> float:
        return float(probs[classes.index(label)]) if label in classes else 0.0

    return {
        "predicted": label_map[pred_cls],
        "negative": round(prob_for(0), 4),
        "neutral": round(prob_for(1), 4),
        "positive": round(prob_for(2), 4),
        "confidence": round(float(probs[pred_idx]), 4),
        "model_version": metadata["sha256"][:12],
    }
```

### Practice

Create a quick Python shell and run:

```python
from src.predict import predict_sentence

predict_sentence("Operating profit increased after strong demand in key markets.")
predict_sentence("Weak demand and higher costs reduced operating margin.")
```

Check:

- Does each result include all three class probabilities?
- Do the probabilities add up to about 1?
- Does each result include a model version?

## Module 11 - Score an Earnings Call Transcript

Add:

```python
def predict_call(transcript: str, ticker: str = "UNKNOWN") -> dict:
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", transcript)
        if len(s.strip()) >= 10
    ]

    if not sentences:
        raise ValueError("Transcript is too short to score.")

    scored = [{**predict_sentence(s), "text": s[:160]} for s in sentences]

    pos = np.mean([row["positive"] for row in scored])
    neg = np.mean([row["negative"] for row in scored])
    score = float(pos - neg)

    direction = "Bullish" if score > 0.15 else "Bearish" if score < -0.15 else "Neutral"

    return {
        "ticker": ticker.upper(),
        "aggregate_sentiment_score": round(score, 4),
        "direction_label": direction,
        "n_sentences": len(scored),
        "positive_sentence_pct": round(sum(r["predicted"] == "positive" for r in scored) / len(scored), 4),
        "negative_sentence_pct": round(sum(r["predicted"] == "negative" for r in scored) / len(scored), 4),
        "top_positive": sorted(scored, key=lambda r: r["positive"], reverse=True)[:3],
        "top_negative": sorted(scored, key=lambda r: r["negative"], reverse=True)[:3],
        "model_version": scored[0]["model_version"],
    }
```

Add batch scoring:

```python
def batch_score(texts: list[str]) -> list[dict]:
    return [predict_sentence(text) | {"text": text[:120]} for text in texts]
```

### Why Transcript Aggregation Is Tricky

The model is trained on sentence-level labels, not full earnings calls. To score a transcript, you split it into sentences, score each sentence, and aggregate the results.

This is useful, but it is not perfect. A real production system might use:

- Speaker labels
- Prepared remarks versus Q&A separation
- Time-aware sentiment changes
- Larger transformer models
- Human review workflows

### Exercise

Improve the transcript scorer by adding:

- `neutral_sentence_pct`
- `average_confidence`
- A warning if fewer than 3 sentences were scored

## Module 12 - Build the FastAPI Service

Create `api/app.py`:

```python
"""FastAPI service for financial sentiment scoring."""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict import SentimentRegistry, batch_score, predict_call, predict_sentence
```

Add logging and startup loading:

```python
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
logger = logging.getLogger("earnings_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        SentimentRegistry.load()
        logger.info("Sentiment model ready.")
    except FileNotFoundError as exc:
        logger.warning(str(exc))
    yield
```

Create the app:

```python
app = FastAPI(
    title="NLP Earnings Sentiment API",
    description="Financial sentence and earnings transcript sentiment scoring.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Add latency middleware:

```python
@app.middleware("http")
async def latency_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Response-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000:.2f}"
    return response
```

### Practice

Explain:

1. Why is `/healthz` different from `/readyz`?
2. Why is request latency useful?
3. Why should API input schemas have minimum and maximum lengths?

## Module 13 - Add API Schemas and Endpoints

Add request and response models:

```python
class SentenceRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=2000)


class SentenceResponse(BaseModel):
    request_id: str
    text: str
    predicted: str
    positive: float
    neutral: float
    negative: float
    confidence: float
    model_version: str
    latency_ms: Optional[float] = None


class CallRequest(BaseModel):
    transcript: str = Field(..., min_length=50)
    ticker: str = Field("UNKNOWN", max_length=10)


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=500)
```

Add operational endpoints:

```python
@app.get("/")
def root():
    return {"service": "NLP Earnings Sentiment API", "version": app.version}


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/readyz")
def ready():
    try:
        _, meta = SentimentRegistry.load()
        metrics = meta.get("metrics", {})
        return {
            "status": "ready",
            "sha": meta.get("sha256", "unknown")[:12],
            "test_accuracy": metrics.get("test_accuracy"),
            "test_macro_f1": metrics.get("test_macro_f1"),
        }
    except Exception as exc:
        raise HTTPException(503, detail=str(exc)) from exc
```

Add model info:

```python
@app.get("/v1/model/info")
def model_info():
    try:
        _, meta = SentimentRegistry.load()
        return meta
    except FileNotFoundError as exc:
        raise HTTPException(404, detail=str(exc)) from exc
```

Add scoring:

```python
@app.post("/v1/sentence", response_model=SentenceResponse)
def score_sentence(req: SentenceRequest):
    t0 = time.perf_counter()

    try:
        result = predict_sentence(req.text)
    except Exception as exc:
        logger.error("Sentence scoring error: %s", exc)
        raise HTTPException(500, detail=str(exc)) from exc

    return SentenceResponse(
        request_id=str(uuid.uuid4()),
        text=req.text[:120] + "..." if len(req.text) > 120 else req.text,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        **result,
    )


@app.post("/v1/call")
def score_call(req: CallRequest):
    t0 = time.perf_counter()

    try:
        result = predict_call(req.transcript, req.ticker)
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc

    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    result["request_id"] = str(uuid.uuid4())
    return result


@app.post("/v1/batch")
def score_batch(req: BatchRequest):
    t0 = time.perf_counter()

    try:
        results = batch_score(req.texts)
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc

    return {
        "request_id": str(uuid.uuid4()),
        "count": len(results),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "results": results,
    }
```

Add model reload:

```python
@app.post("/v1/model/reload")
def reload():
    try:
        _, meta = SentimentRegistry.reload()
        return {"status": "reloaded", "sha": meta.get("sha256", "unknown")[:12]}
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc
```

Run the API:

```bash
uvicorn api.app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

### Exercise

Use the docs page to test:

1. One positive sentence
2. One negative sentence
3. A three-sentence earnings-call transcript
4. A batch of five sentences

Record the outputs in a short project note.

## Module 14 - Add Tests

Create `tests/test_sentiment.py`:

```python
"""Tests for the real-data earnings sentiment project."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "api"))

from train import lm_features
```

Add a data test:

```python
def test_financial_phrasebank_dataset_present():
    path = ROOT / "data" / "financial_phrasebank.csv"
    assert path.exists()

    df = pd.read_csv(path)

    assert {"text", "label", "label_name", "source", "config"}.issubset(df.columns)
    assert set(df["label"].unique()) == {0, 1, 2}
    assert df["source"].eq("Financial PhraseBank").all()
```

Add a feature test:

```python
def test_lexicon_features_are_stable():
    feats = lm_features(pd.Series([
        "strong profit growth improved results",
        "losses declined because demand was weak",
        "",
    ]))

    assert feats.shape == (3, 7)
    assert feats[0, 6] > 0
    assert feats[1, 6] < 0
    assert not np.isnan(feats).any()
```

Add metadata and prediction tests:

```python
def test_model_metadata_has_real_data_metrics():
    meta = json.loads((ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8"))

    assert meta["dataset"] == "Financial PhraseBank"
    assert meta["source_url"].startswith("https://huggingface.co/datasets/")
    assert meta["n_rows"] >= 2000
    assert meta["metrics"]["test_accuracy"] > 0.65
    assert meta["metrics"]["test_macro_f1"] > 0.6
    assert "sha256" in meta


def test_sentence_prediction_contract():
    from predict import predict_sentence

    result = predict_sentence("Operating profit increased after strong demand in key markets.")

    assert result["predicted"] in {"positive", "neutral", "negative"}
    assert 0 <= result["confidence"] <= 1
    assert abs(result["positive"] + result["neutral"] + result["negative"] - 1) < 0.01
    assert len(result["model_version"]) == 12
```

Add API tests:

```python
def test_api_endpoints():
    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)

    assert client.get("/healthz").json()["status"] == "ok"

    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["test_accuracy"] > 0.65

    sentence = client.post(
        "/v1/sentence",
        json={"text": "Operating profit increased after strong demand in key markets."},
    )
    assert sentence.status_code == 200
    assert sentence.json()["predicted"] in {"positive", "neutral", "negative"}
    assert "X-Response-Time-Ms" in sentence.headers

    batch = client.post(
        "/v1/batch",
        json={"texts": ["Revenue improved this quarter.", "Losses increased sharply."]},
    )
    assert batch.status_code == 200
    assert batch.json()["count"] == 2
```

Run:

```bash
pytest tests/ -q
```

### Practice

Add one more test for `/v1/call`.

Your test should verify:

- The ticker is uppercased
- `direction_label` is one of `Bullish`, `Bearish`, or `Neutral`
- The response includes `top_positive` and `top_negative`

## Module 15 - Add Docker Packaging

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python src/download_data.py && python src/train.py

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build:

```bash
docker build -t nlp-earnings-sentiment .
```

Run:

```bash
docker run --rm -p 8000:8000 nlp-earnings-sentiment
```

### Exercise

Answer:

1. Why does the Dockerfile train the model during image build?
2. What is one downside of training during image build?
3. How would production separate training from serving?

## Module 16 - Write the Portfolio README

Your root `README.md` should include:

- Project title
- Business problem
- Dataset source
- Modeling approach
- Metrics table
- API endpoints
- Quickstart commands
- Docker commands
- Project structure
- Governance notes

Example metrics table:

```markdown
| Metric | Value |
| --- | ---: |
| Validation accuracy | 0.9091 |
| Test accuracy | 0.9051 |
| Test macro F1 | 0.8717 |
```

### Practice

Write a short recruiter-facing summary:

```text
Built a production-style financial NLP API that classifies earnings and finance-domain text using real Financial PhraseBank data. The project includes reproducible data ingestion, TF-IDF plus finance lexicon features, model evaluation, model metadata, FastAPI serving, tests, Docker packaging, and governance documentation.
```

Rewrite it in your own words.

## Final Assignment

Build the complete project from scratch, then add one improvement.

Choose one:

- Add a `/v1/compare` endpoint that compares two earnings-call transcripts
- Add neutral sentence percentage and average confidence to transcript scoring
- Add a CLI script that scores a local transcript text file
- Train with `sentences_75agree` and compare metrics against `sentences_allagree`
- Add a simple drift report comparing sentence length and class distribution over time

Your final submission must include:

- Working data download script
- Working training script
- Saved model artifact
- Model metadata JSON
- Model card
- Results dashboard
- FastAPI app
- Tests
- Dockerfile
- README

## Rubric

| Area | Strong Submission |
| --- | --- |
| Data | Uses real Financial PhraseBank data and documents the source |
| Features | Combines TF-IDF with finance-specific lexicon features |
| Modeling | Uses proper train/validation/test split and macro F1 |
| API | Provides health, readiness, model info, sentence, transcript, and batch scoring |
| Testing | Tests data, features, metadata, prediction contracts, and API behavior |
| Governance | Includes model card, intended use, and limitations |
| Portfolio | README explains business value and technical choices clearly |

## Interview Questions

Practice answering these out loud:

1. Why did you use Financial PhraseBank?
2. Why did you choose `sentences_allagree`?
3. What does TF-IDF represent?
4. Why did you add finance lexicon features?
5. Why is macro F1 useful here?
6. What does your model card communicate?
7. How does the API know which model version it is serving?
8. What are the risks of using sentence-level training data for full transcript scoring?
9. How would you monitor this model in production?
10. How would you improve this project with more time?

## Professional Extensions

To make this project even stronger:

- Add MLflow experiment tracking
- Add a transformer baseline using FinBERT
- Add confidence thresholds for human review
- Add drift monitoring for text length, vocabulary, and prediction distribution
- Add structured logging for every prediction request
- Add CI with GitHub Actions
- Add cloud deployment to AWS Lambda, ECS, or SageMaker

## Completion Checklist

You are done when:

- `python src/download_data.py` works
- `python src/train.py` works
- `pytest tests/ -q` passes
- `uvicorn api.app:app --reload` starts the API
- `/docs` shows all endpoints
- `/v1/sentence` returns class probabilities
- `/v1/call` returns aggregate transcript sentiment
- `monitoring/model_card.md` exists
- `monitoring/results_dashboard.png` exists
- Your README explains the project clearly

