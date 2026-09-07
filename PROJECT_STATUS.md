# Project Status: Predicting Household Savings Behavior in Nigeria

## Project Overview
**Objective:** Develop supervised machine learning models to predict whether a Nigerian household has savings (`savings_target`) based on demographic, economic, geographic, asset, income, financial-access, and digital-access characteristics.
**Problem Type:** Binary Classification.
**Target Variable:** `savings_target` (0 = No savings, 1 = Has savings).

## Current State
- **Data Source:** Nigeria General Household Survey, Wave 5.
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

## Pending Roadmap (Next Steps)

### Phase 1: Data Preprocessing
- [ ] **Handle Missing Values:** Apply median imputation for numericals and mode for categoricals.
- [ ] **Feature Transformation:** Apply log transformations to highly skewed variables (earnings, asset values).
- [ ] **Categorical Encoding:** Implement One-Hot Encoding for nominal variables (e.g., State, Geopolitical Zone).
- [ ] **Feature Scaling:** Apply `StandardScaler` to numerical features to ensure model convergence.

### Phase 2: Model Development
- [ ] **Train/Test Split:** Perform a stratified split to maintain target class proportions.
- [ ] **Model Implementation:** Train the following models:
    - Logistic Regression (Baseline)
    - Decision Tree
    - Random Forest
    - XGBoost / Gradient Boosting
- [ ] **Evaluation:** Compare models using Accuracy, Precision, Recall, F1-score, and ROC-AUC.

### Phase 3: Analysis & Conclusion
- [ ] **Feature Importance:** Identify and analyze the most influential predictors.
- [ ] **Financial Inclusion Insights:** Discuss findings in the context of financial access and inclusion in Nigeria.
- [ ] **Final Report:** Document the entire pipeline and results.

## Critical Files
- `13_master_household_ml_dataset.csv`: The master feature set.
- `eda_savings_nigeria.ipynb`: The a-priori analysis used to guide preprocessing.
- `model_development_nigeria.ipynb`: End-to-end preprocessing, model training, cross-validation, evaluation, and feature importance.
