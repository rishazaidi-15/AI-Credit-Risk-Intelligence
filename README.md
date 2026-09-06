# AI-Powered Credit Risk Intelligence Platform

An end-to-end AI-powered credit risk decision-support platform built using the [Home Credit Default Risk dataset](https://www.kaggle.com/competitions/home-credit-default-risk/data) from Kaggle.

The platform combines exploratory data analysis, machine learning, explainable AI, business-readable rules, natural-language data querying, and a Streamlit interface. It was developed for the NeoStats AI Engineer candidate assignment.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Business Problem](#business-problem)
3. [Assignment Coverage](#assignment-coverage)
4. [Architecture](#architecture)
5. [Technology Stack](#technology-stack)
6. [Repository Structure](#repository-structure)
7. [Dataset](#dataset)
8. [Setup and Run Instructions](#setup-and-run-instructions)
9. [Data Understanding and EDA](#data-understanding-and-eda)
10. [Machine Learning Layer](#machine-learning-layer)
11. [Model Results](#model-results)
12. [Explainable AI](#explainable-ai)
13. [Business Rule Derivation](#business-rule-derivation)
14. [Talk-to-Data System](#talk-to-data-system)
15. [Prompt Engineering and Token Optimization](#prompt-engineering-and-token-optimization)
16. [User Interface](#user-interface)
17. [Docker Deployment](#docker-deployment)
18. [Testing](#testing)
19. [Configuration](#configuration)
20. [Known Limitations and Future Improvements](#known-limitations-and-future-improvements)
21. [Responsible Use](#responsible-use)

---

## Project Overview

This project is designed as a complete credit risk decision-support platform rather than only a machine learning prediction script.

Given an applicant's information, the platform can:

1. Predict the probability of loan default.
2. Convert the probability into a LOW, MEDIUM, or HIGH risk band.
3. Explain which factors influenced a prediction.
4. Show global model feature importance.
5. Derive simplified business-readable rules from model behavior.
6. Provide exploratory analysis and data-driven business insights.
7. Allow users to ask questions about the dataset in natural language.
8. Convert supported questions into validated SQL queries.
9. Display the complete workflow through a Streamlit application.

The overall workflow is:

```text
Home Credit Dataset
        |
        v
Data Loading and Validation
        |
        v
EDA and Business Insights
        |
        v
Feature Engineering
        |
        v
Model Training and Comparison
        |
        v
Saved Model Pipeline
        |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
Risk Prediction         Explainability        Business Rules
        |                      |                      |
        +----------------------+----------------------+
                               |
                               v
                       Streamlit Application
                               |
                               v
                     Talk-to-Data SQL Assistant
```

---

## Business Problem

Banks and lending institutions need credit decisions that are accurate, consistent, explainable, and auditable.

A default probability alone is not sufficient for a business user. A useful decision-support system should also help answer questions such as:

- How likely is the applicant to default?
- What risk category does that probability represent?
- Which factors influenced the prediction?
- Which features are important to the model overall?
- Can model behavior be translated into understandable business rules?
- Can business users explore applicant data without writing SQL?

This project addresses those needs by combining machine learning with explainability, data exploration, rule extraction, and natural-language interaction.

---

## Assignment Coverage

The project addresses the major requirements of the NeoStats AI Engineer assignment.

| Assignment Requirement | Implementation |
|---|---|
| Data Understanding and EDA | Dataset summary, data quality analysis, feature categorization, visualizations, and business insights |
| Talk-to-Data | Natural-language question handling with schema-grounded SQL generation and validation |
| Machine Learning | Logistic Regression, Random Forest, and XGBoost comparison |
| Default Prediction | Default probability and LOW, MEDIUM, HIGH risk classification |
| Class Imbalance | Imbalance-aware model configuration without oversampling leakage |
| Explainable AI | SHAP-based global and local explanations |
| Business Rules | Simplified IF-THEN rules derived from model behavior |
| User Interface | Multi-section Streamlit application |
| Database | SQLite applicants table and prediction logging |
| Docker | Dockerfile and Docker Compose configuration |
| Testing | Automated tests for major data, ML, and SQL safety functionality |
| Documentation | Architecture, setup, model rationale, results, prompting approach, limitations, and rule logic |

The assignment expects an end-to-end workflow covering EDA, machine learning, explainability, talk-to-data, business rules, UI, and Docker deployment.

---

## Architecture

```text
application_train.csv
        |
        v
src/data/loader.py
Data Loading and Validation
        |
        +-----------------------------+
        |                             |
        v                             v
src/eda/analysis.py          Column Description Reference
EDA and Data Quality
        |
        v
src/data/feature_engineering.py
Single Feature Engineering Layer
        |
        v
src/ml/train.py
Model Training and Comparison
        |
        v
src/ml/pipeline.py
Preprocessing and Model Pipeline
        |
        v
models/
    credit_risk_pipeline.joblib
    feature_metadata.json
    model_comparison.json
        |
        v
src/ml/predict.py
        |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
src/ml/risk.py       src/explainability/       src/rules/
Risk Bands           shap_explainer.py         rule_derivation.py
        |                      |                      |
        +----------------------+----------------------+
                               |
                               v
                         Streamlit UI
                           app/app.py
                               |
                               v
                     Talk-to-Data System
                               |
                               v
                      SQLite Database
                 data/processed/credit_risk.db
```

### Key Design Decisions

#### One feature engineering layer

Feature cleaning and engineering are defined centrally so that training and inference use the same feature logic.

This reduces the risk of training and inference feature drift.

#### One risk classification function

Probability-to-risk-band logic is centralized.

Configured thresholds are:

```text
Probability < 0.30        LOW
0.30 to < 0.60            MEDIUM
Probability >= 0.60       HIGH
```

#### Model selection based on evidence

The project does not assume that a particular algorithm is the best model.

Candidate models are trained and evaluated before selecting the final model.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.11 |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn |
| Gradient Boosting | XGBoost |
| Explainability | SHAP |
| Visualization | Plotly |
| Database | SQLite, SQLAlchemy |
| SQL Validation | sqlglot |
| LLM Provider | Mock provider by default, Anthropic provider when configured |
| User Interface | Streamlit |
| Environment Management | python-dotenv |
| Testing | pytest |
| Deployment | Docker and Docker Compose |

The project dependencies include pandas, NumPy, scikit-learn, XGBoost, SHAP, Plotly, SQLAlchemy, sqlglot, Anthropic, Streamlit, python-dotenv, and pytest.

---

## Repository Structure

```text
credit_risk_platform/
│
├── app/
│   ├── app.py
│   └── pages/
│
├── data/
│   ├── raw/
│   │   ├── application_train.csv
│   │   └── HomeCredit_columns_description.csv
│   │
│   └── processed/
│       └── credit_risk.db
│
├── models/
│   ├── credit_risk_pipeline.joblib
│   ├── feature_metadata.json
│   └── model_comparison.json
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
│   │   ├── train.py
│   │   ├── pipeline.py
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
│   │   ├── providers/
│   │   ├── sql_validator.py
│   │   └── ...
│   │
│   └── utils/
│       ├── config.py
│       └── logger.py
│
├── tests/
│
├── .dockerignore
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Dataset

The project uses the [Home Credit Default Risk dataset](https://www.kaggle.com/competitions/home-credit-default-risk/data), hosted as a Kaggle competition.

Required files:

```text
data/raw/application_train.csv
data/raw/HomeCredit_columns_description.csv
```

The primary dataset contains applicant-level information and the target column:

```text
TARGET = 0   No default
TARGET = 1   Default
```

The dataset includes demographic, financial, employment, family, housing, and external credit score information.

The column description file is used to improve feature understanding and business-readable explanations.

The NeoStats assignment identifies the Home Credit Default Risk dataset as the source dataset for this project.

---

## Setup and Run Instructions

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd credit_risk_platform
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Create the environment file

```powershell
Copy-Item .env.example .env
```

The default configuration uses:

```env
LLM_PROVIDER=mock
```

This allows the Talk-to-Data module to run without requiring an API key.

### 5. Add the dataset files

Place the required files in:

```text
data/raw/application_train.csv
data/raw/HomeCredit_columns_description.csv
```

### 6. Configure local training

For a lower-memory local training run, set:

```env
DATA_SAMPLE_SIZE=50000
```

Leave it blank to use the full dataset:

```env
DATA_SAMPLE_SIZE=
```

The project supports optional sampling for development because loading the full dataset may require more memory.

### 7. Train the model

```powershell
python -m src.ml.train
```

Training compares candidate models and creates:

```text
models/credit_risk_pipeline.joblib
models/feature_metadata.json
models/model_comparison.json
```

### 8. Initialize and populate the application database

The project database is:

```text
data/processed/credit_risk.db
```

The applicants table contains a business-relevant subset of dataset columns, while the predictions table stores prediction activity.

### 9. Run the application locally

```powershell
streamlit run app/app.py
```

Open:

```text
http://localhost:8501
```

---

## Data Understanding and EDA

The EDA layer is designed to understand the data before model development.

It covers:

- Dataset shape and summary
- Data types
- Missing values
- Target distribution
- Feature categorization
- Financial variables
- Demographic variables
- External credit score features
- Default-related patterns
- Business insights and visualizations

The assignment specifically requires dataset summary, data quality observations, feature categorization, at least five business insights, and supporting visualizations.

Business insights are computed from the available data rather than being hardcoded as fixed conclusions.

---

## Machine Learning Layer

### Objective

Predict the probability that an applicant will default.

### Candidate Models

Three models are trained:

1. Logistic Regression
2. Random Forest
3. XGBoost

### Class Imbalance Strategy

Loan default is the minority class.

Instead of applying oversampling before the train-test split, the models use imbalance-aware settings:

- Logistic Regression uses `class_weight="balanced"`
- Random Forest uses `class_weight="balanced"`
- XGBoost uses `scale_pos_weight`

This approach avoids introducing resampling leakage between training and evaluation data.

### Evaluation Metrics

The project evaluates:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- Confusion matrix
- Inference time per row

Accuracy is not the primary model selection metric because an imbalanced dataset can produce misleadingly high accuracy even when the minority class is poorly identified.

### Model Selection Logic

Models are ranked using:

1. PR-AUC
2. Recall as a tie-breaker
3. Inference speed as a final tie-breaker

PR-AUC is prioritized because the project focuses on identifying the minority default class.

---

## Model Results

The local training run using:

```env
DATA_SAMPLE_SIZE=50000
RANDOM_SEED=42
```

produced the following results.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-score | Inference ms/row |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7479 | 0.2279 | 0.1605 | 0.6832 | 0.2600 | 0.1010 |
| Random Forest | 0.7446 | 0.2124 | 0.2015 | 0.5205 | 0.2906 | 0.9611 |
| XGBoost | 0.7460 | 0.2318 | 0.1898 | 0.6075 | 0.2893 | 0.1889 |

### Selected Model

**XGBoost**

XGBoost was selected because it achieved the highest PR-AUC:

```text
PR-AUC: 0.2318
ROC-AUC: 0.7460
Precision: 0.1898
Recall: 0.6075
F1-score: 0.2893
```

The model selection strategy selected XGBoost based primarily on PR-AUC rather than assuming it would be the best model.

### Probability Distribution Check

For the selected model, predicted probability percentiles on the test set were:

```text
50th percentile: 0.3299
75th percentile: 0.5075
90th percentile: 0.6807
95th percentile: 0.7585
99th percentile: 0.8530
```

These values were logged to provide evidence when interpreting the configured MEDIUM and HIGH risk thresholds.

---

## Explainable AI

The project uses SHAP for model explainability.

Two levels of explanation are supported.

### Global Explainability

Global explanations help identify which features generally influence model predictions across applicants.

This helps answer:

```text
Which variables are most important to the model overall?
```

### Local Explainability

Local explanations focus on an individual prediction.

This helps answer:

```text
Why did this particular applicant receive this risk prediction?
```

Feature contributions are translated into understandable factors so the output is more useful to non-technical users.

The assignment requires at least one explainability method that makes predictions understandable to non-technical users.

---

## Business Rule Derivation

Machine learning models can be difficult to communicate directly to business stakeholders.

The project includes a rule derivation layer that produces simplified IF-THEN style rules intended to approximate patterns in model behavior.

Example format:

```text
IF external credit score is below a learned threshold
AND financial conditions indicate elevated risk
THEN applicant risk may increase
```

These rules are intended for interpretation and communication.

They are not a replacement for the trained predictive model.

The rules are presented with appropriate disclaimers because simplified rules cannot reproduce every interaction learned by the underlying model.

---

## Talk-to-Data System

The Talk-to-Data module allows users to ask questions about the applicant dataset using natural language.

The workflow is:

```text
Natural-Language Question
        |
        v
Provider
        |
        v
SQL Generation
        |
        v
SQL Validation
        |
        v
SQLite Query Execution
        |
        v
Result Limiting
        |
        v
Readable Business Response
```

### Database

SQLite is used for structured data access.

The main table is:

```text
applicants
```

A business-relevant subset of columns is exposed to keep the query schema focused and manageable.

The project also stores prediction records in:

```text
predictions
```

This allows prediction activity to be logged.

### SQL Safety

Generated SQL is validated before execution.

The system is designed for read-only data queries and limits returned rows before sending results further downstream.

This reduces unnecessary data exposure and prevents large query results from being passed through the conversational workflow.

### LLM Providers

The default provider is:

```env
LLM_PROVIDER=mock
```

The mock provider is deterministic and can be used without an API key.

An Anthropic provider can be configured through environment variables when an API key and supported model are available.

---

## Prompt Engineering and Token Optimization

The Talk-to-Data module is designed to reduce hallucination and unnecessary token usage.

Key strategies include:

### Schema Grounding

The provider receives a focused schema description rather than unrestricted database context.

This helps constrain SQL generation to known tables and columns.

### SQL Validation

Generated SQL is validated before execution.

The database execution layer assumes queries have already passed the read-only safety validation step.

### Result Limits

Large result sets are capped before being returned to downstream components.

This reduces unnecessary token usage and keeps responses manageable.

### Conversation Memory Limit

The number of previous conversation turns retained is configurable:

```env
CONVERSATION_MEMORY_TURNS=3
```

This prevents unlimited conversation history from growing the prompt unnecessarily.

### Focused Database Schema

Only a business-relevant subset of applicant columns is exposed to the Talk-to-Data workflow.

This keeps prompts smaller and makes generated SQL easier to validate.

The assignment specifically requests documentation of prompt engineering, token optimization, and hallucination-control approaches.

---

## User Interface

The platform is presented through a Streamlit application.

The interface covers the main platform workflow, including:

- Project overview
- Exploratory data analysis
- Risk prediction
- Explainability
- Business rules
- Talk-to-Data interaction

The UI is intended to demonstrate the system end to end rather than exposing the modules only through Python scripts.

---

## Docker Deployment

The repository includes:

```text
Dockerfile
docker-compose.yml
.dockerignore
.env.example
```

The Docker configuration uses a Python 3.11 environment and runs the Streamlit application on port `8501`.

The Compose configuration:

- Builds the application image
- Loads environment variables from `.env`
- Exposes port `8501`
- Mounts the local `data` directory
- Mounts the local `models` directory

### Before Running Docker

Ensure that:

1. Docker Desktop is running.
2. The Docker Engine is available.
3. `.env` exists.
4. Required dataset files are available in `data/raw`.
5. Model artifacts are available in `models`.

If model artifacts have not been created yet, run:

```powershell
python -m src.ml.train
```

### Build and Run

From the project root:

```powershell
docker compose up --build
```

The application is configured to be reachable at:

```text
http://localhost:8501
```

To stop the application:

```powershell
docker compose down
```

> **Verification note:** Confirm `docker compose up --build` completes without errors and that the application loads correctly at `http://localhost:8501` in your own Docker environment before submission. This README documents the intended Docker configuration; it does not itself constitute a record of a verified successful container run.

The NeoStats assignment requires a Dockerfile, docker-compose.yml, .env.example, and setup instructions for running the application.

---

## Testing

The project includes automated tests covering major functionality across the application.

The local test run completed with:

```text
35 passed, 1 warning
```

Run tests with:

```powershell
pytest
```

The specific cause of the warning was not captured alongside this result and is not described here. Re-run `pytest -v` locally to see the full warning text and its source before including further detail in documentation.

---

## Configuration

Configuration is centralized through environment variables.

Important settings include:

```env
LLM_PROVIDER=mock

RAW_DATA_PATH=data/raw/application_train.csv
COLUMN_DESCRIPTION_PATH=data/raw/HomeCredit_columns_description.csv
PROCESSED_DATA_PATH=data/processed/applicants_processed.csv

DATA_SAMPLE_SIZE=50000
RANDOM_SEED=42

MODEL_DIR=models
MODEL_FILENAME=credit_risk_pipeline.joblib
FEATURE_METADATA_FILENAME=feature_metadata.json

MEDIUM_RISK_THRESHOLD=0.30
HIGH_RISK_THRESHOLD=0.60

SQLITE_DB_PATH=data/processed/credit_risk.db

CONVERSATION_MEMORY_TURNS=3

LOG_LEVEL=INFO
```

For final full-dataset training, `DATA_SAMPLE_SIZE` can be left blank if the local machine has sufficient memory.

---

## Known Limitations and Future Improvements

### 1. Training Data Size

The documented model results were produced using a 50,000-row local sample for memory-efficient development.

Future work should compare results against full-dataset training on hardware with sufficient memory.

### 2. Risk Threshold Calibration

The current LOW, MEDIUM, and HIGH thresholds are configurable.

Future work could calibrate thresholds using business costs, portfolio objectives, and validation data.

### 3. Mock Provider

The default Talk-to-Data provider is deterministic and designed for offline development and testing.

A production deployment could use a real LLM provider with additional evaluation and monitoring.

### 4. Production Security

The project is a demonstration platform and does not include a complete production security architecture such as authentication, authorization, secret management, rate limiting, or audit infrastructure.

### 5. Model Monitoring

Production deployment should include:

- Data drift monitoring
- Prediction drift monitoring
- Performance monitoring
- Retraining policies
- Model versioning
- Alerting

### 6. Fairness and Bias

The project does not claim to be a fairness-audited lending system.

Before real-world lending use, the model would require appropriate fairness analysis, regulatory review, governance, and domain validation.

### 7. Explainability

SHAP explanations describe model behavior.

They should not automatically be interpreted as causal explanations.

---

## Responsible Use

This project is an educational and portfolio decision-support demonstration.

It is not a certified production credit underwriting system.

The output should not be used as the sole basis for real lending decisions without:

- Independent model validation
- Fairness and bias auditing
- Regulatory and legal review
- Security controls
- Governance and approval processes
- Ongoing monitoring
- Human oversight

The purpose of the platform is to demonstrate an end-to-end AI engineering workflow involving data analysis, machine learning, explainability, conversational data access, rule derivation, software engineering, testing, and containerized deployment.