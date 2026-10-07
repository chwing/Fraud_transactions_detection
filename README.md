# Real-Time Financial Fraud Detection

An AI-powered platform that receives banking transactions in real time, scores each one with a
machine learning model (XGBoost), and flags potentially fraudulent transactions with a risk score
in a monitoring dashboard.

The model learns from historical transactions and, above all, from each card's own behaviour:
a transaction is suspicious when it deviates from how that card is normally used (amount, time,
location, frequency, merchant category).

> **Status:** early development — project skeleton and dataset inspection.
> Components are added incrementally, one small layer at a time.

## Target architecture

```text
HISTORICAL DATA → Data Preparation → Feature Engineering → XGBoost → Trained Model
                                                                          │
NEW TRANSACTION → Kafka → Processing / Features → ML Inference ◄──────────┘
                                                       │
                                          Fraud Prediction + Risk Score
                                                       │
                                     ┌─────────────────┴─────────────────┐
                                     ▼                                   ▼
                                PostgreSQL                            FastAPI
                                                                         │
                                                                         ▼
                                                                  React Dashboard
```

Planned stack: Python, Pandas, Scikit-learn, XGBoost, SHAP, Kafka, PostgreSQL, FastAPI,
React + TypeScript, MLflow, Docker Compose.

---

## Dataset

**Credit Card Transactions Fraud Detection Dataset** (Kaggle, `kartik2112/fraud-detection`).

The data is **synthetic**: it was generated with the [Sparkov](https://github.com/namebrandon/Sparkov_Data_Generation)
simulator, which models ~1,000 customers making purchases at ~700 merchants in the United States,
with fraud injected as realistic episodes. No real person or card is represented.

### Files

| File             | Rows      | Frauds | Fraud rate | Period                                 |
|------------------|-----------|--------|------------|----------------------------------------|
| `fraudTrain.csv` | 1,296,675 | 7,506  | 0.58 %     | 2019-01-01 00:00 → 2020-06-21 12:13    |
| `fraudTest.csv`  | 555,719   | 2,145  | 0.39 %     | 2020-06-21 12:14 → 2020-12-31 23:59    |

Key properties:

- **Highly imbalanced** — fewer than 1 transaction in 170 is fraud. A model that always predicts
  "legitimate" is 99.4 % accurate and useless, so accuracy is not a valid metric here.
- **Chronological split** — the test set starts exactly where the training set ends, with no overlap.
  This mirrors production (train on the past, predict the future) and must be preserved: the data
  is never shuffled across this boundary.
- **No missing values** in either file.
- **Cardinality** (train): 983 cards, 693 merchants, 14 categories, 51 states, 894 cities, 494 jobs.

The CSV files are not committed to git. Download them from Kaggle into `data/raw/`.

### Column reference

Each row is one card transaction. *Planned use* is a first proposal and will be finalised when the
transaction schema and features are designed.

**Transaction**

| Column                  | Type     | Meaning                                                                 | Planned use |
|-------------------------|----------|-------------------------------------------------------------------------|-------------|
| *(unnamed first column)*| int      | Row number from the original export                                     | Dropped on load (`index_col=0`) |
| `trans_num`             | string   | Unique transaction identifier (32-char hash)                            | Transaction ID — never a feature |
| `trans_date_trans_time` | datetime | Date and time of the transaction (`YYYY-MM-DD HH:MM:SS`)                | Source of time features (hour, weekday, time since previous transaction) |
| `unix_time`             | int      | Epoch timestamp — **inconsistent**: offset by ~7 years (2,556–2,557 days) from `trans_date_trans_time` | Dropped — redundant and unreliable |
| `amt`                   | float    | Transaction amount in USD (1.00 – 28,948.90 in train)                   | Core feature, raw and relative to the card's usual spend |
| `category`              | string   | Merchant category (14 values, see below)                                | Feature |
| `merchant`              | string   | Merchant name. Every value is prefixed with `fraud_` — a generator artifact present on all rows, **not** a label leak | Candidate (high cardinality: 693) |
| `merch_lat`, `merch_long` | float  | Merchant location (latitude / longitude)                                | Distance between cardholder and merchant |
| `is_fraud`              | int (0/1)| **Target label** — 1 if the transaction is fraudulent                   | Target |

**Card and cardholder**

| Column          | Type     | Meaning                                                     | Planned use |
|-----------------|----------|-------------------------------------------------------------|-------------|
| `cc_num`        | int      | Credit card number — identifies the card / customer         | Grouping key for per-card history — **never a feature** (the model would memorise cards) |
| `first`, `last` | string   | Cardholder first and last name                              | Excluded (PII) |
| `gender`        | string   | Cardholder gender (`F` / `M`)                               | Under discussion (fairness) |
| `dob`           | date     | Cardholder date of birth (1924 – 2005)                      | Age at transaction time — under discussion (fairness) |
| `job`           | string   | Cardholder occupation (494 values)                          | Likely excluded (high cardinality, weak signal) |
| `street`        | string   | Cardholder street address                                   | Excluded (PII) |
| `city`, `state`, `zip` | string / int | Cardholder home city, US state, ZIP code           | Mostly excluded; location is captured by coordinates |
| `lat`, `long`   | float    | Cardholder home location (latitude / longitude)             | Distance between cardholder and merchant |
| `city_pop`      | int      | Population of the cardholder's city                         | Candidate (urban vs rural) |

**Merchant categories (`category`)**

`entertainment`, `food_dining`, `gas_transport`, `grocery_net`, `grocery_pos`, `health_fitness`,
`home`, `kids_pets`, `misc_net`, `misc_pos`, `personal_care`, `shopping_net`, `shopping_pos`, `travel`

The suffix tells the channel: `_net` = online (card not present), `_pos` = in store (point of sale).

### Known limitations

- **Synthetic data** — patterns are cleaner than real-world fraud; results will look better than
  they would on real bank data.
- **No device / channel field** beyond the `_net` / `_pos` category suffix.
- **Mostly fixed customer base** — 908 of the 924 test cards also appear in train; only 16 are new.
  Per-card history is rich, but new-card (cold start) behaviour is barely represented.

---

## Project structure

```text
anomly_bank_detection/
├── README.md
├── pyproject.toml           # dependencies + project metadata
├── .gitignore               # ignores data/, .venv/, models/, .env
├── data/
│   ├── raw/                 # original dataset, never modified (not in git)
│   └── processed/           # derived data (not in git)
├── notebooks/               # exploration only, no logic that matters
├── src/
│   └── fraud_detection/     # the real code, as an installable package
│       ├── __init__.py
│       └── config.py        # paths and settings in one place
└── tests/
```

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# check
python -c "from fraud_detection import config; print(config.RAW_DATA_DIR)"
pytest
```

Then download `fraudTrain.csv` and `fraudTest.csv` into `data/raw/`.
