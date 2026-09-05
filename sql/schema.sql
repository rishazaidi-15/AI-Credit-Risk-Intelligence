-- AI-Powered Credit Risk Intelligence Platform -- SQLite schema
-- This file documents the schema created programmatically by
-- src/database/database.py (init_db). It is a reference, not something you
-- need to run manually -- the app creates these tables automatically.

-- Processed applicant records (loaded from the cleaned/engineered dataset).
-- One row per applicant; mirrors the same feature set the ML pipeline uses,
-- so the Talk-to-Data chatbot can answer questions using real applicant data.
CREATE TABLE IF NOT EXISTS applicants (
    "SK_ID_CURR" INTEGER PRIMARY KEY,
    "TARGET" INTEGER,
    "NAME_CONTRACT_TYPE" TEXT,
    "CODE_GENDER" TEXT,
    "FLAG_OWN_CAR" TEXT,
    "FLAG_OWN_REALTY" TEXT,
    "CNT_CHILDREN" INTEGER,
    "AMT_INCOME_TOTAL" REAL,
    "AMT_CREDIT" REAL,
    "AMT_ANNUITY" REAL,
    "AMT_GOODS_PRICE" REAL,
    "NAME_TYPE_SUITE" TEXT,
    "NAME_INCOME_TYPE" TEXT,
    "NAME_EDUCATION_TYPE" TEXT,
    "NAME_FAMILY_STATUS" TEXT,
    "NAME_HOUSING_TYPE" TEXT,
    "DAYS_BIRTH" INTEGER,
    "DAYS_EMPLOYED" INTEGER,
    "OCCUPATION_TYPE" TEXT,
    "ORGANIZATION_TYPE" TEXT,
    "CNT_FAM_MEMBERS" REAL,
    "EXT_SOURCE_1" REAL,
    "EXT_SOURCE_2" REAL,
    "EXT_SOURCE_3" REAL,
    "REGION_RATING_CLIENT" INTEGER,
    "REGION_POPULATION_RELATIVE" REAL
    -- Additional raw columns are loaded dynamically from the processed
    -- dataset at ingestion time (see database.load_applicants_table); this
    -- lists the core columns the chatbot's schema-grounded prompt relies on.
);

-- Stored prediction results, one row per prediction request served by the app.
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sk_id_curr INTEGER,
    probability REAL NOT NULL,
    risk_band TEXT NOT NULL,
    model_name TEXT,
    created_at_utc TEXT NOT NULL
);
