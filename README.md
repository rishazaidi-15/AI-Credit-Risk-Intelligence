# AI-Powered Credit Risk Intelligence Platform

An end-to-end, explainable AI platform for credit risk assessment, built on
the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
Kaggle dataset. It covers the full workflow from raw data to a decision-support
UI: data understanding, machine learning, explainability, business rule
derivation, a natural-language chatbot over the data, and Docker deployment.

## 1. Project Overview

Given an applicant's data, the platform predicts the probability that they
will default on a loan, converts that probability into a business-readable
risk band (LOW / MEDIUM / HIGH), explains *why* the model reached that
conclusion, surfaces simplified decision rules that approximate the model's
behavior, and lets a business user ask plain-English questions about the
underlying dataset through an AI assistant.

## 2. Business Problem

Banks need faster, more consistent, and more explainable credit decisions.
A model's raw probability output is not enough on its own -- underwriters
and analysts need to know *why* an applicant was classified as high risk,
whether that reasoning holds up, and how to explore the broader dataset to
sanity-check policy decisions. This platform is built around that full
decision-support loop, not just a prediction endpoint.

## 3. Solution Overview

| Capability | How it's addressed |
|---|---|
| Risk identification | A trained classifier scores every applicant with a default probability. |
| Risk scoring & bands | Probability is converted to LOW / MEDIUM / HIGH via a single, centrally configured threshold function. |
| Explainability | SHAP-based global and local explanations, translated into business language. |
| Policy intelligence | A surrogate decision tree extracts simplified, auditable IF-THEN rules. |
| Business intelligence | A schema-grounded, SQL-validated chatbot answers natural-language questions about the data. |

## 4. Architecture

```
Home Credit Dataset (application_train.csv)
        |
        v
Data Loading & Validation (src/data/loader.py)
        |
        v
EDA / Data Quality (src/eda/analysis.py)
        |
        v
Feature Engineering (src/data/feature_engineering.py)  <-- single feature schema, used by training AND inference
        |
        v
Model Training & Comparison (src/ml/train.py, pipeline.py, evaluate.py)
        |
        v
Saved Pipeline (models/credit_risk_pipeline.joblib) + Feature Metadata (models/feature_metadata.json)
        |
        v
Inference (src/ml/predict.py)  ---------------------------+
        |                                                  |
        v                                                  v
Risk Classification (src/ml/risk.py)          Explainability (src/explainability/shap_explainer.py)
        |                                                  |
        v                                                  v
Business Rules (src/rules/rule_derivation.py)   SQLite (src/database/database.py)
        |                                                  |
        +-------------------+  +---------------------------+
                             |  |
                             v  v
                   Streamlit UI (app/app.py + app/pages/*)
                             |
                             v
              Talk-to-Data AI Assistant (src/talk_to_data/*)
```

**Key design decision -- one feature pipeline, everywhere:** `src/data/feature_engineering.py`
defines the entire modeling feature schema and all cleaning/engineering logic
exactly once. Training, single-applicant prediction, SHAP explanation, and
rule derivation all call the same functions, so there is no way for
training-time and inference-time feature logic to drift apart.

**Key design decision -- one risk-band function, everywhere:** `src/ml/risk.py`
is the only place probability-to-band thresholds are applied. The UI, the
database layer, and the rule-derivation module all import from it rather
than re-implementing the threshold logic.

## 5. Key Features

- Configurable-sampling dataset loader with clean, non-crashing error handling
- EDA with 5+ genuine, data-computed business insights and data quality analysis
- 3-way model comparison (Logistic Regression, Random Forest, XGBoost) selected by evidence, not assumption
- SHAP-based global and local explainability translated into plain-language factors
- Auditable, disclaimer-labeled business rules extracted from model behavior
- SQLite-backed structured data layer with prediction logging
- Schema-grounded, SQL-validated, hallucination-resistant NL-to-SQL chatbot
- Pluggable LLM provider: free, deterministic `mock` provider by default, real `anthropic` provider when configured
- 6-page Streamlit UI: Home/Overview, EDA, Prediction, Explainability, Rules, AI Assistant
- Dockerized deployment
- 35 automated tests covering data, ML, and SQL safety

## 6. Technology Stack

| Layer | Technology |
|---|---|
| Data | pandas, numpy |
| ML | scikit-learn (Logistic Regression, Random Forest), XGBoost |
| Explainability | SHAP |
| Database | SQLite via SQLAlchemy |
| NL-to-SQL safety | sqlglot (AST-based SQL validation) |
| LLM | Anthropic API (`anthropic` SDK), with a free offline mock provider |
| UI | Streamlit |
| Config | python-dotenv |
| Testing | pytest |
| Deployment | Docker, Docker Compose |

## 7. Project Structure

```
credit_risk_platform/
├── app/
│   ├── app.py                  # Streamlit entry point + navigation + dashboard
│   ├── pages/                  # eda.py, prediction.py, explainability.py, rules.py, chatbot.py
│   └── components/
│       └── ui_components.py    # risk badges, error banners
├── data/
│   ├── raw/                    # place application_train.csv + column description here (gitignored)
│   └── processed/               # SQLite DB lives here (gitignored)
├── models/                     # trained pipeline + metadata (gitignored)
├── notebooks/                  # optional EDA notebook
├── sql/
│   └── schema.sql              # reference schema documentation
├── src/
│   ├── data/                   # loader.py, feature_engineering.py
│   ├── eda/                    # analysis.py
│   ├── ml/                     # pipeline.py, train.py, evaluate.py, predict.py, risk.py
│   ├── explainability/         # shap_explainer.py
│   ├── rules/                  # rule_derivation.py
│   ├── database/               # database.py
│   ├── talk_to_data/           # nl_to_sql.py, sql_validator.py, prompt_templates.py,
│   │                           #   conversation_memory.py, llm_provider.py, providers/
│   └── utils/                  # config.py, logger.py
├── tests/                      # test_preprocessing.py, test_ml.py, test_sql_validator.py,
│                               #   generate_sample_data.py (dev-only smoke-test tool)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## 8. Dataset Information

This project uses the **Home Credit Default Risk** Kaggle competition dataset.
Specifically:

- `application_train.csv` -- the primary modeling dataset (has `TARGET`)
- `HomeCredit_columns_description.csv` -- official column descriptions, used
  for business-readable feature categorization

Of the ~120 raw columns, 31 were selected (base features) spanning
demographics, financials, employment, external credit scores, credit bureau
inquiries, and social-circle indicators -- chosen for interpretability and
business relevance rather than exhaustively using every column. See
`src/data/feature_engineering.py` for the exact list and rationale.

**Known data quality issue handled explicitly:** `DAYS_EMPLOYED` uses the
sentinel value `365243` to represent "not currently employed." This is
detected generically by the Phase 2 suspicious-value analysis and explicitly
cleaned in `feature_engineering.py` (converted to `NaN` plus a new
`IS_RETIRED_OR_UNEMPLOYED_FLAG` feature) rather than being silently treated
as a real ~1000-year tenure value.

## 9. Dataset Setup

1. Download from: https://www.kaggle.com/competitions/home-credit-default-risk/data
2. Place the two files at:
   ```
   data/raw/application_train.csv
   data/raw/HomeCredit_columns_description.csv
   ```
   (Paths are configurable via `.env` if you'd rather store them elsewhere.)
3. **Do not commit these files to Git** -- `data/raw/` is already gitignored.

**Working with a subset for faster iteration:** set `DATA_SAMPLE_SIZE` in
`.env` to a row count (e.g. `50000`). Leave it blank to use the full dataset
-- recommended for the final trained model.

**No real dataset yet?** Run `python tests/generate_sample_data.py` to
generate a small synthetic file at `tests/fixtures/sample_application_train.csv`
with the correct schema, purely so you can smoke-test the pipeline while you
wait on the Kaggle download. This is clearly separated from the real
pipeline and must never be used as the actual training/production dataset.

## 10. Installation (Windows)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item ".env.example" ".env"
```

## 11. Configuring `.env`

Key settings (see `.env.example` for the full list with comments):

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `mock` (free, offline, deterministic) or `anthropic` (real LLM) |
| `ANTHROPIC_API_KEY` | required only when `LLM_PROVIDER=anthropic` |
| `ANTHROPIC_MODEL` | model string, e.g. `claude-sonnet-4-5` |
| `RAW_DATA_PATH` / `COLUMN_DESCRIPTION_PATH` | dataset file locations |
| `DATA_SAMPLE_SIZE` | optional row cap for fast dev iteration (blank = full dataset) |
| `MEDIUM_RISK_THRESHOLD` / `HIGH_RISK_THRESHOLD` | risk band boundaries (default 0.30 / 0.60) |
| `SQLITE_DB_PATH` | where the chatbot's database lives |
| `CONVERSATION_MEMORY_TURNS` | how many chat turns the assistant remembers |

The app never crashes due to missing configuration: `settings.validate()` is
checked on the Home page and surfaces clean warnings instead.

## 12. Training the Model

Requires the real dataset in place (see Section 9).

```powershell
python -m src.ml.train
```

This trains and compares Logistic Regression, Random Forest, and XGBoost,
selects the best by PR-AUC (tie-broken by minority-class recall, then
inference speed), and saves:

- `models/credit_risk_pipeline.joblib` -- the winning fitted pipeline
- `models/feature_metadata.json` -- feature lists, chosen model, test metrics
- `models/model_comparison.json` -- all 3 candidates' metrics, for transparency

**Class imbalance strategy:** the target is heavily imbalanced (most
applicants repay). Logistic Regression and Random Forest use
`class_weight='balanced'`; XGBoost uses `scale_pos_weight`. This avoids any
oversampling-related train/test leakage risk entirely.

**Evaluation metrics:** ROC-AUC and PR-AUC are the primary decision metrics
(PR-AUC is more informative under imbalance than ROC-AUC alone), alongside
precision, recall, F1, and the confusion matrix. Accuracy is intentionally
not used for model selection.

**Risk threshold justification:** training logs the chosen model's
predicted-probability percentiles on the held-out test set, so the default
0.30 / 0.60 thresholds can be checked against the actual score distribution
rather than assumed to be correct out of the box.

## 13. Running the Application

```powershell
streamlit run app/app.py
```

Open the URL Streamlit prints (typically http://localhost:8501).

## 14. Running Tests

```powershell
pytest tests/ -v
```

35 tests cover: dataset loading and validation, feature engineering
consistency (including sparse/partial records and train/inference
determinism), risk band thresholds, model training/evaluation/selection, and
SQL validator safety (valid queries pass; DROP/DELETE/UPDATE/INSERT/ALTER/
CREATE/ATTACH/PRAGMA and unknown tables/columns are all rejected).

## 15. The AI Assistant (Talk-to-Data)

Flow: **User question -> LLM generates SQL -> SQL validator -> SQL execution
-> LLM generates business-readable answer -> response shown to user**,
with the generated SQL visible in an expandable section for transparency.

**Provider abstraction:** `src/talk_to_data/llm_provider.py` defines a common
interface. With `LLM_PROVIDER=mock` (the default), a deterministic,
keyword-based provider handles the 5 required query patterns (aggregation,
filtering, grouping, comparison, ranking) with zero API cost -- useful for
development and grading without Anthropic billing configured. With
`LLM_PROVIDER=anthropic` and a valid `ANTHROPIC_API_KEY`, the real Anthropic
provider is used instead, with no changes needed anywhere else in the app.

**Hallucination controls:**
- The LLM only ever sees the actual database schema (`get_schema_description()`) -- it cannot invent tables/columns.
- All generated SQL is parsed and validated by `sqlglot` before execution: SELECT-only, known tables/columns only, single-statement only.
- The answer-generation prompt explicitly instructs the model to use only the returned SQL result and to say so plainly if the data is insufficient.

**Conversation memory:** capped at `CONVERSATION_MEMORY_TURNS` (default 3),
session-scoped, never sends unbounded history to the LLM.

**Error handling:** database and LLM provider errors are caught and shown as
clean, chat-appropriate messages (e.g. "The AI service is temporarily
unavailable") -- raw exceptions are logged internally only.

## 16. Explainability

`src/explainability/shap_explainer.py` computes SHAP values from the actual
trained pipeline:

- **Global:** mean absolute SHAP value across a sample of applicants, shown as a bar chart of top contributing factors.
- **Local:** per-applicant SHAP values for the last prediction made, split into "top factors increasing risk" and "top factors reducing risk," with raw feature names translated into business language (e.g. `EXT_SOURCE_2` -> "External credit score #2").

If SHAP computation fails for any reason, the UI shows a clean fallback
message rather than a fabricated explanation -- explanations are never
faked.

## 17. Business Rules

`src/rules/rule_derivation.py` trains a shallow (depth-3) surrogate decision
tree on the model's own predicted risk bands and extracts IF-THEN rules from
its leaves, each annotated with supporting sample count and purity. Every
rule is shown with an explicit disclaimer: these are **model-derived
insights, not official credit policy**, and should support -- not replace --
human underwriting judgment.

## 18. Sample Outputs

Example rule (actual output from a trained model):
```
IF EXT_SOURCE_MEAN > 0.56 AND DAYS_REGISTRATION <= -6413.0 AND AGE_YEARS > 40.63
THEN applicant is more likely to fall into LOW RISK
(based on 447 similar applicants, 80% consistent)
```

Example chatbot exchange:
```
Q: Which occupation groups have the highest default rate?
Generated SQL: SELECT OCCUPATION_TYPE, AVG(TARGET) AS default_rate FROM applicants
               WHERE OCCUPATION_TYPE IS NOT NULL GROUP BY OCCUPATION_TYPE
               ORDER BY default_rate DESC LIMIT 5
A: Based on the available data: [{'OCCUPATION_TYPE': 'Sales staff', 'default_rate': 0.046}, ...]
```

## 19. Limitations / Responsible Use

- This platform is a **decision-support demonstration**, not a certified
  credit underwriting system. It must not be used for real lending decisions
  without proper validation, fairness auditing, and regulatory review.
- The dataset reflects historical patterns and may encode existing societal
  or institutional biases; model predictions should be interpreted with that
  in mind, particularly for protected-characteristic-adjacent features.
- Business rules are derived from a simplified surrogate model and
  approximate -- not exactly reproduce -- the full model's behavior.
- The mock LLM provider is a deterministic stand-in for demonstration
  purposes; it is not a real language model and will not generalize beyond
  its 5 documented query patterns.
- SHAP explanations describe model behavior, not causal real-world
  relationships -- a feature "increasing predicted risk" is not the same as
  that feature causing default.

## 20. Future Improvements

- Incorporate auxiliary Home Credit tables (`bureau.csv`, `previous_application.csv`) for richer features
- Add fairness/bias auditing across protected characteristics
- Support LIME as an alternative/complementary explainability method
- Add a query-regeneration retry loop when generated SQL fails validation
- Add authentication and multi-user prediction history

## 21. Major Design Decisions

- **One feature pipeline, everywhere** (Section 4) -- prevents train/inference drift.
- **One risk-band function, everywhere** (Section 4) -- prevents contradictory risk calculations.
- **Provider abstraction for the LLM** -- develop and grade for free with `mock`; upgrade to real Anthropic-powered NL-to-SQL by setting two env vars, no code changes.
- **Evidence-based model selection** -- 3 candidates trained and compared on held-out data; the "best" model is whichever wins on PR-AUC/recall/speed, not assumed in advance.
- **Shared preprocessing across all 3 candidate models** -- standard-scaling numeric features doesn't materially hurt tree models, and one preprocessing path is simpler and safer than maintaining per-model preprocessing.
- **Rules trained on predicted risk bands, not raw TARGET** -- the rule module explains *model behavior*, which is a distinct (and clearly labeled) concept from ground-truth default outcomes.
- **Synthetic data is strictly a dev tool** -- `tests/generate_sample_data.py` is used only for fast local smoke-testing and is never the dataset the shipped model is trained on.
