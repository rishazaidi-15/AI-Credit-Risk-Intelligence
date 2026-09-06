# AI-Powered Credit Risk Intelligence Platform

*An end-to-end, explainable AI decision-support platform for credit risk assessment — combining machine learning, business intelligence, SHAP explainability, model-derived rules, secure natural-language data querying, and Dockerized deployment.*

Built on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) Kaggle dataset.

---

## Overview

This project is not simply a machine learning prediction script — it is designed as a complete **credit risk decision-support platform**.

Given an applicant's information, the platform:

1. Predicts the probability that the applicant may default on a loan.
2. Converts that probability into a business-readable risk category — **LOW**, **MEDIUM**, or **HIGH**.
3. Explains *why* the model produced that prediction, both globally and for the specific applicant.
4. Derives simplified, auditable IF-THEN rules that approximate the model's behavior.
5. Provides exploratory data analysis and genuine, data-computed business insights.
6. Lets users ask natural-language questions about the dataset, safely converted into validated SQL.
7. Runs through a Streamlit UI and deploys via Docker Compose.

The guiding workflow:

```
Raw Data → Data Validation → EDA & Business Insights → Feature Engineering
→ Model Training & Comparison → Risk Prediction → Explainability
→ Model-Derived Rules → Database → AI/Natural-Language Data Assistant
→ Streamlit UI → Docker Deployment
```

## Why This Project — The Business Problem

Banks and lending institutions need decisions that are faster, more consistent, data-driven, explainable, and auditable. A raw machine learning probability alone is insufficient — a business user or underwriter also needs to know:

- How likely is default, and what risk category does that represent?
- Which applicant characteristics influenced this specific prediction?
- Which factors generally influence the model across all applicants?
- Can simplified, business-readable rules approximate the model's behavior?
- Can an analyst explore the underlying data without writing SQL by hand?

This project is built around that full decision-support loop, not just a model endpoint.

> **Responsible-use framing:** This is a decision-support demonstration and educational/portfolio project. It is **not** a certified production credit underwriting system and must not be used for real lending decisions without appropriate model validation, fairness and bias auditing, regulatory review, security controls, governance, and monitoring.

## Key Features

### 1. Data Loading and Validation
Configurable dataset paths, optional row sampling for faster development, clean and specific error handling, and structural dataset validation before anything downstream runs.

### 2. Exploratory Data Analysis and Business Insights
A dedicated EDA page provides a dataset summary, data quality report, target/default distribution, and multiple business insights computed directly from the data — including the relationship between external credit scores and default behavior. Insights are surfaced from the data itself, not hardcoded conclusions.

### 3. Machine Learning Model Comparison
Three candidate models are trained and compared — **Logistic Regression**, **Random Forest**, and **XGBoost** — and the best one is selected using evidence rather than assuming a winner in advance.

Selection approach:
1. **PR-AUC** (primary)
2. Minority-class **recall** (tie-break)
3. Inference speed (final tie-break)

Evaluation metrics: ROC-AUC, PR-AUC, precision, recall, F1-score, and confusion matrix. Accuracy is intentionally **not** the primary selection metric, because the target is heavily imbalanced — a model that simply predicts "no default" for everyone would score high on accuracy while being useless.

**Why PR-AUC matters here:** when the positive class (default) is rare, PR-AUC is far more informative than ROC-AUC because it focuses on the precision/recall trade-off for the minority class specifically — the class a credit risk platform actually cares about getting right.

### 4. Class Imbalance Handling
Most applicants do not default, so the target is heavily imbalanced. This is handled through class weighting rather than oversampling:

| Model | Imbalance Strategy |
|---|---|
| Logistic Regression | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight` |

Avoiding oversampling techniques (like SMOTE) also avoids the risk of oversampling-related train/test leakage.

### 5. Credit Risk Prediction
Users enter applicant information across demographics, financial details, and employment/credit signals. The model outputs a default probability, which is mapped to a risk band using centrally configured thresholds:

- `MEDIUM_RISK_THRESHOLD = 0.30`
- `HIGH_RISK_THRESHOLD = 0.60`

*Example applicant prediction observed during development:* **MEDIUM RISK — 47.3% default probability.** This is an illustrative single-applicant result, not a model performance metric.

### 6. Explainability
Powered by SHAP:

- **Global explainability** — mean absolute SHAP values across a sample of applicants, showing which features most influence the model overall.
- **Local explainability** — factors increasing and reducing predicted risk for one specific applicant's prediction.

Technical feature names are translated into business-readable labels (e.g. `EXT_SOURCE_2` → *"External credit score #2"*). If SHAP computation fails for any reason, the UI shows a clean fallback message — the platform never fabricates an explanation.

> SHAP describes model behavior and feature association with a prediction. It does **not** establish causal relationships.

### 7. Model-Derived Decision Rules
A shallow surrogate decision tree (depth 3) is trained to approximate the main model's predicted risk bands, and readable IF-THEN rules are extracted from its leaves. Each rule includes its risk category, conditions, supporting sample count, and consistency/purity.

*Example rule observed during development:*

```
IF EXT_SOURCE_MEAN > 0.56
AND DAYS_REGISTRATION <= -6413.0
AND AGE_YEARS > 40.63
THEN applicant is more likely to fall into LOW RISK
(based on 447 similar applicants, 80% consistent)
```

These are **model-derived insights**, not official credit policy. They approximate — but do not exactly reproduce — the full ML model's behavior, and are meant to support, not replace, human underwriting judgment.

### 8. SQLite Data Layer
A SQLite database (via SQLAlchemy) supports structured querying for the AI assistant and prediction workflows, including prediction logging as part of the structured data design.

### 9. AI-Powered Talk-to-Data Assistant
Ask questions about the applicant dataset in plain English, for example:

- *What is the average income of applicants?*
- *How many applicants have children?*
- *Which income category has the highest default rate?*
- *Compare the default rate between employed and unemployed applicants.*
- *Which occupation groups have the highest default rate?*

*Example query result observed during development:*

| Rank | Occupation Group | Observed Default Rate |
|---|---|---|
| 1 | Low-skill Laborers | 15.57% |
| 2 | Waiters/barmen staff | 11.84% |
| 3 | IT staff | 11.54% |
| 4 | Drivers | 11.15% |
| 5 | Security staff | 10.69% |

Assistant flow:

```
User Question → LLM generates SQL → SQL Validation → SQL Execution
→ Business-readable answer formatting → Response displayed to user
```

Generated SQL can be viewed in an expandable section for transparency.

### 10. Pluggable LLM Provider
A common interface (`src/talk_to_data/llm_provider.py`) supports two interchangeable providers, switchable purely through configuration:

| Provider | Behavior |
|---|---|
| `mock` (default) | Free, offline, deterministic, keyword-based. Supports the 5 required query patterns: aggregation, filtering, grouping, comparison, and ranking. Useful for development, demonstration, testing, and grading without API cost. |
| `anthropic` | Activated with `LLM_PROVIDER=anthropic` and a valid `ANTHROPIC_API_KEY`. |

The mock provider is a deterministic development/demo fallback — it is **not** a real LLM and does not generalize beyond its documented query patterns.

### 11. SQL Safety and Hallucination Controls
This is one of the strongest engineering aspects of the platform. SQL is validated using **sqlglot**, with AST-based parsing rather than simple keyword matching — meaning the query's actual structure is checked, not just scanned for suspicious words.

Enforced controls:
- SELECT-only queries
- Only known tables
- Only known columns
- Single-statement queries only

Rejected operations include `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `CREATE`, `ATTACH`, and `PRAGMA`, as well as any unknown table or column reference.

Hallucination controls:
1. The LLM only ever receives the actual database schema.
2. All generated SQL is parsed and validated before execution.
3. Generated SQL is restricted to safe, read-only operations.
4. Answer generation uses only the returned SQL results.
5. If the data is insufficient to answer a question, the assistant states that rather than inventing a response.
6. Conversation history is capped (`CONVERSATION_MEMORY_TURNS`, default `3`) — session-scoped and bounded, never unbounded history.

Errors from the database or LLM provider are shown to the user as clean, readable messages, while raw technical exceptions are logged internally only.

### 12. Streamlit User Interface
Six main pages: **Home/Overview**, **EDA & Business Insights**, **Risk Prediction**, **Explainability**, **Decision Rules**, and **AI Assistant** — covering dataset summaries, business insights, interactive prediction, global/local model explainability, model-derived rules, and conversational dataset querying.

## Architecture

```
Home Credit Dataset
(application_train.csv)
        |
        v
Data Loading & Validation
(src/data/loader.py)
        |
        v
EDA / Data Quality Analysis
(src/eda/analysis.py)
        |
        v
Feature Engineering
(src/data/feature_engineering.py)
        |
        v
Model Training & Comparison
(src/ml/train.py, pipeline.py, evaluate.py)
        |
        v
Saved Model Pipeline                Feature Metadata
(models/credit_risk_pipeline.joblib)  (models/feature_metadata.json)
        |
        v
Inference
(src/ml/predict.py)
        |
        +-----------------------------+
        |                             |
        v                             v
Risk Classification              Explainability
(src/ml/risk.py)                 (src/explainability/
        |                          shap_explainer.py)
        |                             |
        v                             |
Model-Derived Rules                   |
(src/rules/rule_derivation.py)        |
        |                             |
        +--------------+--------------+
                       |
                       v
                Streamlit Application
                  (app/app.py)
                       |
        +--------------+--------------+
        |                             |
        v                             v
    SQLite Data Layer          Talk-to-Data AI Assistant
(src/database/database.py)      (src/talk_to_data/*)
```

### Key Architectural Design Decisions

**One feature pipeline, everywhere.** `src/data/feature_engineering.py` defines the feature schema and all cleaning/engineering logic centrally. Training, single-applicant prediction, SHAP explainability, and rule derivation all use this same logic — preventing training/inference feature drift, one of the most common sources of silent bugs in ML systems.

**One risk-band function, everywhere.** `src/ml/risk.py` is the single place where a default probability becomes a risk band. The UI, database layer, and rule-derivation component all reuse this logic instead of independently reimplementing thresholds — preventing inconsistent or contradictory risk classifications across the application.

## Technology Stack

| Category | Technologies |
|---|---|
| Data | pandas, numpy |
| Machine Learning | scikit-learn (Logistic Regression, Random Forest), XGBoost |
| Explainability | SHAP |
| Database | SQLite, SQLAlchemy |
| NL-to-SQL Safety | sqlglot (AST-based validation) |
| LLM | Anthropic API (`anthropic` SDK), offline deterministic mock provider |
| UI | Streamlit |
| Configuration | python-dotenv |
| Testing | pytest |
| Deployment | Docker, Docker Compose |

## Project Structure

```
credit_risk_platform/
│
├── app/
│   ├── app.py
│   ├── pages/
│   │   ├── eda.py
│   │   ├── prediction.py
│   │   ├── explainability.py
│   │   ├── rules.py
│   │   └── chatbot.py
│   └── components/
│       └── ui_components.py
│
├── data/
│   ├── raw/              # gitignored — see Dataset Setup
│   └── processed/        # gitignored where appropriate
│
├── models/                # trained artifacts, gitignored depending on repo configuration
├── notebooks/
│
├── sql/
│   └── schema.sql
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── feature_engineering.py
│   │
│   ├── eda/
│   │   └── analysis.py
│   │
│   ├── ml/
│   │   ├── pipeline.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── risk.py
│   │
│   ├── explainability/
│   │   └── shap_explainer.py
│   │
│   ├── rules/
│   │   └── rule_derivation.py
│   │
│   ├── database/
│   │   └── database.py
│   │
│   ├── talk_to_data/
│   │   ├── nl_to_sql.py
│   │   ├── sql_validator.py
│   │   ├── prompt_templates.py
│   │   ├── conversation_memory.py
│   │   ├── llm_provider.py
│   │   ├── query_runner.py
│   │   └── providers/
│   │       ├── mock_provider.py
│   │       └── anthropic_provider.py
│   │
│   └── utils/
│       ├── config.py
│       └── logger.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_ml.py
│   ├── test_sql_validator.py
│   └── generate_sample_data.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .dockerignore
└── README.md
```

## Dataset

**Source:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk)

Primary files used:

| File | Purpose |
|---|---|
| `application_train.csv` | Primary modeling dataset; contains the `TARGET` column |
| `HomeCredit_columns_description.csv` | Official column descriptions, used for business-readable feature categorization |

The raw dataset files are **not committed to Git**. Expected locations (configurable via environment variables):

```
data/raw/application_train.csv
data/raw/HomeCredit_columns_description.csv
```

For faster development, `DATA_SAMPLE_SIZE` can limit the number of rows loaded (e.g. `DATA_SAMPLE_SIZE=50000`); leave it blank to use the full dataset.

The application was demonstrated with 50,000 rows and the full 122 raw columns (104 numerical, 16 categorical). From this, approximately **31 base features** were selected for interpretability and business relevance, spanning demographics, financial information, employment information, external credit scores, credit bureau inquiries, and social-circle indicators. Feature engineering then transforms these selected base features into the final modeling schema — during development this produced **39 modeling features** (27 numeric, 12 categorical). These two numbers are not contradictory: 31 is the curated starting set, 39 is the result after feature engineering (e.g. derived ratios, encoded indicators).

### Data Quality Handling

A significant, known data-quality issue in this dataset is handled explicitly: `DAYS_EMPLOYED` uses the sentinel value **365243** to represent applicants who are not currently employed — not a literal employment duration of roughly 1,000 years.

The platform:
1. Detects suspicious placeholder/sentinel values automatically during data quality analysis.
2. Explicitly handles `DAYS_EMPLOYED` during feature engineering.
3. Converts the sentinel value to `NaN`.
4. Creates an `IS_RETIRED_OR_UNEMPLOYED_FLAG` feature to preserve the signal without misrepresenting it as a genuine tenure value.

This flag indicates the sentinel condition was present in the source data — it is a data quality signal, not a confirmed employment status.

## Installation and Local Setup (Windows)

```powershell
git clone <REPOSITORY_URL>
cd credit_risk_platform

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt

Copy-Item ".env.example" ".env"
```

Replace `<REPOSITORY_URL>` with your actual repository URL.

### Environment Configuration

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `mock` or `anthropic` |
| `ANTHROPIC_API_KEY` | Required only when `LLM_PROVIDER=anthropic` |
| `ANTHROPIC_MODEL` | Anthropic model identifier |
| `RAW_DATA_PATH` / `COLUMN_DESCRIPTION_PATH` | Dataset file locations |
| `DATA_SAMPLE_SIZE` | Optional row limit for faster development |
| `MEDIUM_RISK_THRESHOLD` | Default `0.30` |
| `HIGH_RISK_THRESHOLD` | Default `0.60` |
| `SQLITE_DB_PATH` | SQLite database file location |
| `CONVERSATION_MEMORY_TURNS` | Remembered conversation turns, default `3` |

### Dataset Setup

Place the dataset files at:

```
data/raw/application_train.csv
data/raw/HomeCredit_columns_description.csv
```

## Model Training

```powershell
python -m src.ml.train
```

Training process:
1. Load and validate data.
2. Apply feature engineering.
3. Split data appropriately.
4. Train three candidate models — Logistic Regression, Random Forest, XGBoost.
5. Evaluate each candidate.
6. Select the best model, primarily by PR-AUC, with minority-class recall and inference speed as tie-breaks.
7. Save the winning pipeline and supporting metadata.

Saved artifacts:

| File | Contents |
|---|---|
| `models/credit_risk_pipeline.joblib` | The winning fitted preprocessing + model pipeline |
| `models/feature_metadata.json` | Feature information, selected model, and evaluation metrics |
| `models/model_comparison.json` | Full comparison across all candidate models, for transparency |

### Risk Thresholds

Risk bands use centrally configured thresholds (defaults shown below; configurable via environment variables):

| Probability | Risk Band |
|---|---|
| `< 0.30` | LOW |
| `0.30` – `< 0.60` | MEDIUM |
| `>= 0.60` | HIGH |

Training also logs the predicted-probability percentiles on the held-out test set, so these thresholds can be checked against the model's actual score distribution rather than assumed blindly.

## Running the Application

```powershell
streamlit run app/app.py
```

The app runs by default at `http://localhost:8501`.

## Docker Deployment

```powershell
docker compose up --build
```

Then open `http://localhost:8501`.

To stop:

```powershell
docker compose down
```

Configuration summary:

| Setting | Value |
|---|---|
| Service | `credit-risk-app` |
| Container name | `credit_risk_platform` |
| Port mapping | `8501:8501` |
| Volumes | `./data:/app/data`, `./models:/app/models` |
| Restart policy | `unless-stopped` |

The Dockerfile exposes port 8501 and runs:

```
streamlit run app/app.py --server.port=8501 --server.address=0.0.0.0
```

The dataset and trained model artifacts are mounted as volumes rather than baked into the image, keeping the image lightweight and letting the container use whichever dataset/model version is present on the host without a rebuild.

## Testing and Validation

```powershell
pytest tests/ -v
```

**Result:** `35 passed, 1 warning in 21.90s`

The single warning is a SciPy/scikit-learn dependency deprecation notice — not a test failure.

Test coverage includes:
- Dataset loading and validation
- Feature engineering consistency, including sparse/partial records
- Training/inference determinism
- Risk band thresholds
- Model training, evaluation, and selection
- SQL validator safety — valid read queries pass; unsafe operations (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `CREATE`, `ATTACH`, `PRAGMA`) and references to unknown tables/columns are rejected

## Application Screenshots

<!-- Add screenshot: assets/images/eda.png -->
**EDA & Business Insights**

<!-- Add screenshot: assets/images/risk_prediction.png -->
**Credit Risk Prediction**

<!-- Add screenshot: assets/images/explainability.png -->
**Prediction Explainability**

<!-- Add screenshot: assets/images/decision_rules.png -->
**Model-Derived Decision Rules**

<!-- Add screenshot: assets/images/ai_assistant.png -->
**AI-Powered Data Assistant**

<!-- Add screenshot: assets/images/docker_desktop.png -->
**Docker Deployment / Docker Desktop**

## Key Engineering Decisions

| Decision | Why It Matters |
|---|---|
| One feature pipeline everywhere | Prevents training/inference feature drift |
| One risk-band function everywhere | Prevents inconsistent or contradictory risk labels |
| Evidence-based model selection | Three candidates are compared rather than assuming a winner |
| Imbalance-aware evaluation | PR-AUC and recall matter more than accuracy for this problem |
| Shared preprocessing across models | Reduces maintenance complexity, keeps candidates comparable |
| Rules explain model behavior, not policy | Surrogate rules are trained on the model's predicted risk bands, not presented as lending policy |
| Pluggable LLM provider | Mock provider enables free, deterministic development; Anthropic is enabled purely through configuration |
| SQL validated before execution | The LLM never receives unrestricted database access |
| Explanations are never fabricated | If SHAP fails, the UI shows a fallback rather than inventing a reason |
| Synthetic data is development-only | `tests/generate_sample_data.py` supports smoke testing and must never be treated as the production/training dataset |

## Limitations and Responsible AI

1. **Decision-support demonstration only** — not certified for real-world lending decisions.
2. **Historical data bias** — the dataset may reflect historical, social, or institutional bias.
3. **Fairness auditing** is not yet implemented as a full production-grade governance layer.
4. **SHAP is not causal explanation** — a feature influencing the model's output does not mean it causes default.
5. **Surrogate rules approximate** model behavior; they do not exactly reproduce the full model.
6. **The mock provider is deterministic** and limited to its documented query patterns — it is not a general-purpose language model.
7. **Feature scope** — the model currently uses selected features from the main application dataset rather than incorporating auxiliary Home Credit tables.

## Future Improvements

- Incorporate auxiliary Home Credit tables such as `bureau.csv` and `previous_application.csv`
- Add fairness and bias auditing
- Add LIME as an additional explainability option
- Add SQL regeneration/retry when generated SQL fails validation
- Add authentication
- Add multi-user prediction history
- Add experiment tracking / model versioning
- Add production monitoring
- Add CI/CD
- Add a dedicated API layer for deployment

## Project Summary

This platform demonstrates a complete, decision-support-oriented approach to credit risk modeling: validated data ingestion, genuine data-driven EDA, evidence-based model comparison under class imbalance, centralized and consistent risk classification, SHAP-based explainability with no fabricated outputs, auditable model-derived rules clearly separated from policy, a safety-first natural-language data assistant with AST-based SQL validation, and a Dockerized Streamlit interface tying it all together — built to be transparent about what it does, how it works, and where its limitations lie.