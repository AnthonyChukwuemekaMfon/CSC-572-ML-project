# Project Status: Predicting Household Savings Behavior in Nigeria

## Project Overview
**Objective:** Develop supervised machine learning models to predict whether a Nigerian household has savings (`savings_target`) based on demographic, economic, geographic, asset, income, financial-access, and digital-access characteristics.
**Problem Type:** Binary Classification.
**Target Variable:** `savings_target` (0 = No savings, 1 = Has savings).

## Current State
- **Data Source:** Nigeria General Household Survey, Panel Wave 5 (2023/2024) (National Bureau of Statistics / World Bank).
- **Current Dataset:** `13_master_household_ml_dataset.csv` (4,771 households × 42 columns).
- **Completed Work:** 
    - Raw Stata files converted to CSV and aggregated to household level.
    - **Exploratory Data Analysis (EDA)** is complete. The results are documented in `eda_savings_nigeria.ipynb`.
    - Environment issues (NumPy 2.x compatibility) have been resolved.

## EDA Findings Summary
- **Target Distribution:** Analysis of class imbalance has been performed.
- **Key Predictors Identified:**
    - **Financial Access:** Bank access, mobile money, and assisted banking are critical predictors.
    - **Economics:** Income sources and total earnings show strong associations with savings.
    - **Demographics:** Household size, head age, and urban/rural status are relevant.
    - **Assets:** Total asset value and variety of assets owned are strong indicators.
- **Data Quality:**
    - Identified variables with missing values requiring imputation.
    - Identified highly skewed numerical variables (earnings, asset values) requiring transformation.
    - Identified features for removal (e.g., `hhid`, duplicate indicators).

## Roadmap & Completion Status

### Phase 1: Data Preprocessing (COMPLETED)
- [x] **Handle Missing Values:** Applied median imputation for numericals and mode for categoricals.
- [x] **Feature Transformation:** Applied log transformations (`log1p`) to highly skewed variables (earnings, asset values, household size).
- [x] **Categorical Encoding:** Implemented One-Hot Encoding for nominal variables (`zone_code`, `urban_rural_code`, `head_sex_code`).
- [x] **Feature Scaling:** Applied `StandardScaler` strictly fitted on training data to prevent leakage.

### Phase 2: Model Development (COMPLETED)
- [x] **Train/Test Split:** Performed stratified 80/20 train/test split maintaining target class proportions.
- [x] **Model Implementation:** Trained four candidate models:
    - Logistic Regression (Baseline)
    - Decision Tree Classifier
    - Random Forest Classifier
    - XGBoost Classifier
- [x] **Cross-Validation & Evaluation:** Conducted 5-Fold Stratified CV and hold-out test evaluation using Accuracy, Precision, Recall, F1-score, and ROC-AUC.

### Phase 3: Analysis & Conclusion (COMPLETED)
- [x] **Feature Importance:** Analyzed Logistic Regression odds ratios, Random Forest Gini importance, and XGBoost Gain importance.
- [x] **Financial Inclusion Insights:** Evaluated institutional proximity (`any_bank_access`, `any_assisted_banking`, `any_mobile_money_access`) and regional divides.
- [x] **Final Report:** Documented end-to-end technical methodology and policy recommendations in `FINAL_REPORT.md`.

## Deliverables & Critical Files
- `13_master_household_ml_dataset.csv`: Master household feature set.
- `eda_savings_nigeria.ipynb`: Exploratory data analysis notebook.
- `model_development_nigeria.ipynb`: Preprocessing, model training, cross-validation, and evaluation notebook.
- `model_evaluation_metrics.csv`: Test set performance metrics across all models.
- `model_feature_importances.csv`: Coefficients, odds ratios, and tree feature importances.
- `FINAL_REPORT.md`: Comprehensive academic project report and policy recommendations.
- `app.py`: Interactive Streamlit Web Application (Simulator & Real-Time Household Predictor).


