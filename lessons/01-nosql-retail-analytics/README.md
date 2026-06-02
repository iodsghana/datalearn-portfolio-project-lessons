# Lesson 01: Build a NoSQL Retail Analytics Project

## What You Will Build

In this lesson, you will build a complete NoSQL analytics project using real retail transaction data.

By the end, you will have:

- Downloaded a real external dataset
- Cleaned transaction data with Python
- Converted flat transaction rows into MongoDB-style documents
- Created nested order documents with embedded line items
- Built customer RFM segments
- Built product performance documents
- Loaded documents into MongoDB
- Created indexes for common query patterns
- Built a FastAPI analytics service
- Written tests
- Created a clean GitHub-ready project

You are not building a toy script. You are building a small data product.

## Dataset

Dataset: UCI Online Retail Dataset  
Source: https://archive.ics.uci.edu/dataset/352/online+retail

This dataset contains transactions from a UK-based online retail store between December 2010 and December 2011. Each row is a product line item from an invoice.

Important columns:

| Column | Meaning |
| --- | --- |
| `InvoiceNo` | Invoice number. If it starts with `C`, it is a cancellation. |
| `StockCode` | Product code |
| `Description` | Product name |
| `Quantity` | Number of units bought |
| `InvoiceDate` | Date and time of transaction |
| `UnitPrice` | Product price |
| `CustomerID` | Customer identifier |
| `Country` | Customer country |

## Why This Is a NoSQL Project

In a relational database, you normally split data into tables:

- invoices
- invoice line items
- customers
- products

In MongoDB, we often store related data together. For this project, each order document will contain its line items inside the same document.

Example:

```json
{
  "invoice_no": "536365",
  "customer_id": "17850",
  "country": "United Kingdom",
  "order_value": 139.12,
  "line_items": [
    {
      "stock_code": "85123A",
      "description": "WHITE HANGING HEART T-LIGHT HOLDER",
      "quantity": 6,
      "unit_price": 2.55,
      "line_revenue": 15.30
    }
  ]
}
```

This is called an embedded document pattern.

## Before You Start

Install these tools:

- Python 3.11 or newer
- Git
- VS Code
- Docker Desktop, optional but recommended for MongoDB

Create a project folder:

```bash
mkdir nosql-retail-analytics
cd nosql-retail-analytics
```

Create the folder structure:

```text
nosql-retail-analytics/
  api/
  data/
  docker/
  docs/
  src/
  tests/
```

## Module 1: Create the Environment

### Explanation

A virtual environment keeps this project's packages separate from other Python projects.

### Steps

```bash
python -m venv .venv
.venv\Scripts\activate
```

Create `requirements.txt`:

```txt
pandas>=2.3,<3
numpy>=2.2,<3
openpyxl>=3.1,<4
fastapi>=0.115,<1
uvicorn[standard]>=0.34,<1
pydantic>=2.10,<3
pymongo>=4.10,<5
pytest>=8.3,<9
httpx>=0.28,<1
```

Install packages:

```bash
pip install -r requirements.txt
```

### Exercise

Run:

```bash
python -c "import pandas, fastapi, pymongo; print('ready')"
```

Expected output:

```text
ready
```

### Checkpoint Question

Why do we use a virtual environment instead of installing everything globally?

## Module 2: Download and Clean the Real Dataset

### Explanation

The UCI dataset is distributed as a ZIP file containing an Excel workbook. We will download it, extract it, read it with pandas, clean missing values, and create a revenue column.

Create `src/download_data.py`:

```python
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

UCI_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
ZIP_PATH = RAW_DIR / "online_retail.zip"
XLSX_PATH = RAW_DIR / "Online Retail.xlsx"
CLEAN_PATH = DATA_DIR / "online_retail_clean.csv"


def download():
    if not ZIP_PATH.exists():
        urlretrieve(UCI_URL, ZIP_PATH)
    if not XLSX_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as archive:
            archive.extract("Online Retail.xlsx", RAW_DIR)
    return XLSX_PATH


def clean():
    xlsx = download()
    df = pd.read_excel(xlsx, engine="openpyxl")
    df = df.rename(columns={
        "InvoiceNo": "invoice_no",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "UnitPrice": "unit_price",
        "CustomerID": "customer_id",
        "Country": "country",
    })
    df = df.dropna(subset=["invoice_no", "stock_code", "description", "customer_id"])
    df["invoice_no"] = df["invoice_no"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    df["customer_id"] = df["customer_id"].astype(int).astype(str)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df = df.dropna(subset=["quantity", "unit_price"])
    df["is_cancelled"] = df["invoice_no"].str.startswith("C")
    df["line_revenue"] = df["quantity"] * df["unit_price"]
    df.to_csv(CLEAN_PATH, index=False)
    return df


if __name__ == "__main__":
    df = clean()
    print(f"Wrote {len(df):,} cleaned rows")
```

Run:

```bash
python src/download_data.py
```

Expected result:

```text
Wrote 406,829 cleaned rows
```

### Practice Task

Open the cleaned CSV and answer:

1. How many countries are in the data?
2. What is the earliest invoice date?
3. How many rows are cancellations?

Hint:

```python
import pandas as pd
df = pd.read_csv("data/online_retail_clean.csv")
print(df["country"].nunique())
```

### Checkpoint Question

Why do we create `line_revenue` instead of only keeping `quantity` and `unit_price`?

## Module 3: Design the Document Model

### Explanation

Before writing code, decide what your MongoDB collections should look like.

We will create three collections:

| Collection | Purpose |
| --- | --- |
| `orders` | One invoice per document, with embedded line items |
| `customers` | One customer per document, with RFM metrics |
| `products` | One product per document, with sales metrics |

This is an important database design decision.

### Exercise

Write one sentence explaining why `line_items` should be embedded inside `orders`.

Example answer:

```text
Line items belong to one invoice and are usually read together with that invoice, so embedding them makes order lookup simpler.
```

## Module 4: Build Order Documents

### Explanation

The source data has one row per product line item. We need one document per invoice.

Create `src/build_documents.py` and start with:

```python
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CLEAN_PATH = DATA_DIR / "online_retail_clean.csv"
ORDERS_PATH = DATA_DIR / "orders_documents.jsonl"


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def build_orders():
    df = pd.read_csv(CLEAN_PATH, parse_dates=["invoice_date"])
    sales = df[(~df["is_cancelled"]) & (df["quantity"] > 0) & (df["unit_price"] > 0)]

    orders = []
    for invoice_no, group in sales.groupby("invoice_no", sort=False):
        first = group.iloc[0]
        line_items = []

        for _, row in group.iterrows():
            line_items.append({
                "stock_code": row["stock_code"],
                "description": row["description"],
                "quantity": int(row["quantity"]),
                "unit_price": round(float(row["unit_price"]), 2),
                "line_revenue": round(float(row["line_revenue"]), 2),
            })

        orders.append({
            "_id": invoice_no,
            "invoice_no": invoice_no,
            "customer_id": first["customer_id"],
            "invoice_date": first["invoice_date"].isoformat(),
            "country": first["country"],
            "n_items": int(group["quantity"].sum()),
            "n_distinct_products": int(group["stock_code"].nunique()),
            "order_value": round(float(group["line_revenue"].sum()), 2),
            "line_items": line_items,
        })

    write_jsonl(ORDERS_PATH, orders)
    return orders


if __name__ == "__main__":
    orders = build_orders()
    print(f"Wrote {len(orders):,} order documents")
```

Run:

```bash
python src/build_documents.py
```

Expected result:

```text
Wrote 18,532 order documents
```

### Practice Task

Open `data/orders_documents.jsonl` and inspect the first line. Answer:

1. Does it contain an embedded `line_items` list?
2. How many products are inside that first order?
3. What is the order value?

### Common Mistake

If you get this error:

```text
FileNotFoundError: online_retail_clean.csv
```

You skipped Module 2. Run:

```bash
python src/download_data.py
```

## Module 5: Build Customer RFM Documents

### Explanation

RFM stands for:

- Recency: how recently the customer bought
- Frequency: how often the customer bought
- Monetary: how much the customer spent

Add this function to `src/build_documents.py`:

```python
CUSTOMERS_PATH = DATA_DIR / "customers_documents.jsonl"


def rfm_segment(recency, frequency, monetary):
    if frequency >= 20 and monetary >= 2500 and recency <= 60:
        return "Champions"
    if frequency >= 10 and monetary >= 1000:
        return "Loyal"
    if recency > 180 and monetary >= 500:
        return "At Risk"
    if frequency <= 2:
        return "New or Low Activity"
    return "Core"


def build_customers():
    df = pd.read_csv(CLEAN_PATH, parse_dates=["invoice_date"])
    sales = df[(~df["is_cancelled"]) & (df["quantity"] > 0) & (df["unit_price"] > 0)]
    reference_date = sales["invoice_date"].max() + pd.Timedelta(days=1)

    invoice_values = sales.groupby(["customer_id", "invoice_no"])["line_revenue"].sum().reset_index()
    avg_order_value = invoice_values.groupby("customer_id")["line_revenue"].mean()

    grouped = sales.groupby("customer_id").agg(
        country=("country", lambda x: x.mode().iat[0]),
        first_order=("invoice_date", "min"),
        last_order=("invoice_date", "max"),
        frequency=("invoice_no", "nunique"),
        total_revenue=("line_revenue", "sum"),
    )

    customers = []
    for customer_id, row in grouped.iterrows():
        recency = int((reference_date - row["last_order"]).days)
        frequency = int(row["frequency"])
        total_revenue = float(row["total_revenue"])
        customers.append({
            "_id": customer_id,
            "customer_id": customer_id,
            "country": row["country"],
            "first_order": row["first_order"].isoformat(),
            "last_order": row["last_order"].isoformat(),
            "recency_days": recency,
            "frequency": frequency,
            "total_revenue": round(total_revenue, 2),
            "avg_order_value": round(float(avg_order_value.loc[customer_id]), 2),
            "segment": rfm_segment(recency, frequency, total_revenue),
        })

    write_jsonl(CUSTOMERS_PATH, customers)
    return customers
```

Update the bottom of the file:

```python
if __name__ == "__main__":
    orders = build_orders()
    customers = build_customers()
    print(f"Wrote {len(orders):,} order documents")
    print(f"Wrote {len(customers):,} customer documents")
```

Run:

```bash
python src/build_documents.py
```

Expected result:

```text
Wrote 18,532 order documents
Wrote 4,338 customer documents
```

### Practice Task

Find how many customers are in each segment.

Starter code:

```python
import json
from collections import Counter

rows = []
with open("data/customers_documents.jsonl", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

print(Counter(row["segment"] for row in rows))
```

### Checkpoint Question

Why might a marketing team care about `At Risk` customers?

## Module 6: Build Product Documents

### Explanation

Product documents help answer merchandising questions:

- Which products generate the most revenue?
- Which products sell in the most countries?
- Which products appear in many orders?

Add this function:

```python
PRODUCTS_PATH = DATA_DIR / "products_documents.jsonl"


def build_products():
    df = pd.read_csv(CLEAN_PATH, parse_dates=["invoice_date"])
    sales = df[(~df["is_cancelled"]) & (df["quantity"] > 0) & (df["unit_price"] > 0)]

    grouped = sales.groupby(["stock_code", "description"]).agg(
        units_sold=("quantity", "sum"),
        revenue=("line_revenue", "sum"),
        n_orders=("invoice_no", "nunique"),
        countries=("country", "nunique"),
    ).reset_index()

    products = []
    for _, row in grouped.iterrows():
        products.append({
            "_id": row["stock_code"],
            "stock_code": row["stock_code"],
            "description": row["description"],
            "units_sold": int(row["units_sold"]),
            "revenue": round(float(row["revenue"]), 2),
            "n_orders": int(row["n_orders"]),
            "countries_sold": int(row["countries"]),
        })

    write_jsonl(PRODUCTS_PATH, products)
    return products
```

Update the script to run all three builders.

### Exercise

Sort the products by revenue and print the top 10.

### Expected Result

You should produce more than 3,000 product documents.

## Module 7: Save Dataset Metrics

### Explanation

A professional project should tell the reader what data it used and what was created.

Add:

```python
METRICS_PATH = DATA_DIR / "dataset_metrics.json"


def save_metrics(df, sales, orders, customers, products):
    metrics = {
        "source": "UCI Machine Learning Repository - Online Retail",
        "source_url": "https://archive.ics.uci.edu/dataset/352/online+retail",
        "transaction_rows": int(len(df)),
        "sales_rows": int(len(sales)),
        "orders": int(len(orders)),
        "customers": int(len(customers)),
        "products": int(len(products)),
        "countries": int(sales["country"].nunique()),
        "total_revenue": round(float(sales["line_revenue"].sum()), 2),
        "date_min": sales["invoice_date"].min().date().isoformat(),
        "date_max": sales["invoice_date"].max().date().isoformat(),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
```

### Practice Task

Add one more metric:

```text
average_order_value
```

Hint: use total revenue divided by number of orders.

## Module 8: Load Documents Into MongoDB

### Explanation

MongoDB stores data in collections. We will load:

- `orders`
- `customers`
- `products`
- `dataset_metadata`

Create `src/load_mongo.py`:

```python
import json
import os
from pathlib import Path

from pymongo import ASCENDING, DESCENDING, MongoClient, ReplaceOne

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("MONGO_DB", "retail_analytics")


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def upsert_many(collection, docs):
    operations = [ReplaceOne({"_id": doc["_id"]}, doc, upsert=True) for doc in docs]
    if operations:
        collection.bulk_write(operations, ordered=False)


def load():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    orders = read_jsonl(DATA_DIR / "orders_documents.jsonl")
    customers = read_jsonl(DATA_DIR / "customers_documents.jsonl")
    products = read_jsonl(DATA_DIR / "products_documents.jsonl")
    metadata = json.loads((DATA_DIR / "dataset_metrics.json").read_text())

    upsert_many(db.orders, orders)
    upsert_many(db.customers, customers)
    upsert_many(db.products, products)

    db.orders.create_index([("customer_id", ASCENDING), ("invoice_date", DESCENDING)])
    db.orders.create_index([("country", ASCENDING), ("order_value", DESCENDING)])
    db.orders.create_index([("line_items.stock_code", ASCENDING)])
    db.customers.create_index([("segment", ASCENDING), ("total_revenue", DESCENDING)])
    db.products.create_index([("revenue", DESCENDING)])

    db.dataset_metadata.replace_one(
        {"_id": "online_retail"},
        {"_id": "online_retail", **metadata},
        upsert=True,
    )
    client.close()


if __name__ == "__main__":
    load()
    print("MongoDB load complete")
```

### Run MongoDB

If you have Docker:

```bash
docker run --name retail-mongo -p 27017:27017 -d mongo:7
python src/load_mongo.py
```

### Exercise

Explain what this index supports:

```python
db.orders.create_index([("country", ASCENDING), ("order_value", DESCENDING)])
```

## Module 9: Build the FastAPI Service

### Explanation

An API lets other people query your analytics without opening Python or MongoDB.

Create `api/app.py`:

```python
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pymongo import MongoClient

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.environ.get("MONGO_DB", "retail_analytics")


def get_db():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    return client[MONGO_DB]


app = FastAPI(title="NoSQL Retail Analytics API")


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/v1/metadata")
def metadata():
    meta = get_db().dataset_metadata.find_one({"_id": "online_retail"})
    if not meta:
        raise HTTPException(404, detail="Dataset metadata not loaded")
    meta.pop("_id", None)
    return meta


@app.get("/v1/revenue/by-country")
def revenue_by_country(limit: int = Query(10, ge=1, le=50)):
    pipeline = [
        {"$group": {
            "_id": "$country",
            "orders": {"$sum": 1},
            "revenue": {"$sum": "$order_value"},
            "avg_order_value": {"$avg": "$order_value"},
        }},
        {"$sort": {"revenue": -1}},
        {"$limit": limit},
        {"$project": {
            "_id": 0,
            "country": "$_id",
            "orders": 1,
            "revenue": {"$round": ["$revenue", 2]},
            "avg_order_value": {"$round": ["$avg_order_value", 2]},
        }},
    ]
    rows = list(get_db().orders.aggregate(pipeline))
    return {"rows": len(rows), "data": rows}
```

Run:

```bash
uvicorn api.app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

### Practice Task

Add an endpoint:

```text
GET /v1/products/top
```

It should return products sorted by revenue.

## Module 10: Write Tests

### Explanation

Tests prove the project works and help you avoid breaking it later.

Create `tests/test_nosql_analytics.py`:

```python
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_real_clean_dataset_present():
    path = ROOT / "data" / "online_retail_clean.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) > 400_000
    assert df["country"].nunique() >= 30


def test_order_documents_are_nested():
    orders = read_jsonl(ROOT / "data" / "orders_documents.jsonl")
    assert len(orders) > 15_000
    assert isinstance(orders[0]["line_items"], list)
    assert "stock_code" in orders[0]["line_items"][0]


def test_metrics_file():
    meta = json.loads((ROOT / "data" / "dataset_metrics.json").read_text())
    assert meta["source"] == "UCI Machine Learning Repository - Online Retail"
    assert meta["orders"] > 15_000
    assert meta["customers"] > 4_000
```

Run:

```bash
pytest tests/ -q
```

### Exercise

Add a test that checks product documents exist and contain `revenue`.

## Module 11: Add Docker Compose

### Explanation

Docker Compose lets you run MongoDB and the API together.

Create `docker/docker-compose.yml`:

```yaml
services:
  mongo:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.api
    environment:
      MONGO_URI: mongodb://mongo:27017
      MONGO_DB: retail_analytics
    ports:
      - "8000:8000"
    depends_on:
      - mongo

volumes:
  mongo_data:
```

Create `docker/Dockerfile.api`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Run:

```bash
docker compose -f docker/docker-compose.yml up --build
```

### Practice Task

Add a `loader` service that runs:

```bash
python src/load_mongo.py
```

before the API starts.

## Module 12: Write the README

Your README should include:

- Project title
- Dataset source
- What the project does
- Document collections
- How to run the project
- API endpoints
- Tests
- Limitations

### Exercise

Write a short section called:

```text
Why MongoDB?
```

Explain why orders and line items are stored together.

## Final Assignment

Extend the project with one new analytics feature.

Choose one:

1. Add `GET /v1/products/search?keyword=heart`
2. Add `GET /v1/customers/at-risk`
3. Add `GET /v1/revenue/monthly`
4. Add a `returns` collection using cancelled invoices

Your final assignment must include:

- Code
- One test
- README update
- Screenshot or copied output from the API docs

## Final Rubric

| Area | Excellent |
| --- | --- |
| Data ingestion | Downloads and cleans the real UCI data correctly |
| Document modeling | Orders have embedded line items; customers/products are useful analytical documents |
| MongoDB | Loader works and indexes match query patterns |
| API | Endpoints return clear JSON and support filters |
| Tests | Tests check data, documents, and API behavior |
| README | Clear enough for another student to run the project |
| Explanation | Student can explain why NoSQL was used and what tradeoffs were made |

## Interview Practice

Answer these aloud:

1. Why did you embed line items inside order documents?
2. What queries do your indexes support?
3. What is one disadvantage of this NoSQL design?
4. How would you scale this if the dataset had 100 million orders?
5. How would you monitor whether the API is healthy?

## Finished Project Reference

The finished project version is available here:

https://github.com/iodsghana/NoSQL--database-level-project
