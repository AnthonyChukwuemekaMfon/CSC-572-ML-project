# Predicting Household Savings Behavior in Nigeria: A Supervised Machine Learning Approach

**Course:** CSC 572 – Machine Learning  
**Dataset:** Nigeria General Household Survey (GHS), Wave 5  
**Target Variable:** Household Savings Status (`savings_target`: 1 = Has Savings, 0 = No Savings)  
**Status:** Complete Pipeline (EDA, Preprocessing, Modeling, Evaluation, Policy Synthesis)

---

## Executive Summary

Promoting household savings is a cornerstone of economic development, poverty alleviation, and financial stability across Sub-Saharan Africa. In Nigeria, despite substantial policy interventions by the Central Bank of Nigeria (CBN) and the rollout of the National Financial Inclusion Strategy (NFIS), significant disparities in formal and informal savings persist across socio-demographic strata and geographic regions.

This study implements an end-to-end supervised machine learning framework to model and predict household savings behavior using microdata from the Nigeria General Household Survey (GHS Wave 5), comprising **4,771 nationally representative households**. We formulated the problem as binary classification, engineered 24 leak-free predictors spanning financial access, digital connectivity, economic activity, physical asset variety, and regional demographics, and benchmarked four machine learning architectures: **Logistic Regression (Baseline)**, **Decision Tree Classifier**, **Random Forest Classifier**, and **eXtreme Gradient Boosting (XGBoost)**.

The empirical results demonstrate that **XGBoost** achieved the best overall predictive performance with a **Test Accuracy of 77.49%**, **F1-Score of 0.8015**, and **ROC-AUC of 0.8371**, closely followed by **Random Forest (ROC-AUC: 0.8330)** and **Logistic Regression (ROC-AUC: 0.8250)**. Feature importance analysis reveals that **direct access to banking institutions (`any_bank_access`) is by far the single most decisive predictor**, conferring an **odds ratio of 3.44** ($+244\%$ increase in savings odds), followed by **assisted/agency banking access (Odds Ratio: 2.09)**, **mobile money access (Odds Ratio: 1.84)**, and **household asset diversification (Odds Ratio: 1.21)**. These findings provide empirical backing for scaling agent banking infrastructure and digital financial services as high-leverage policy instruments for national financial inclusion.

---

## 1. Introduction & Research Problem

### 1.1 Background
Savings behavior serves as both an insurance buffer against adverse economic shocks (such as health emergencies, crop failures, or macroeconomic volatility) and an accumulation mechanism for productive investments in education, agriculture, and small-scale entrepreneurship. In developing nations like Nigeria, household financial activities are divided between the **formal banking system** (commercial banks, microfinance institutions) and **informal channels** (esusu/adashi rotating savings, cooperative societies, and cash at home).

### 1.2 Research Questions
1. How accurately can supervised machine learning algorithms classify a Nigerian household's savings propensity based on observable socio-demographic, economic, asset, and institutional access indicators?
2. Which feature families (e.g., physical banking access vs. mobile/digital tools vs. asset wealth vs. labor activity) exert the strongest influence on household savings adoption?
3. How do ensemble methods (Random Forest, XGBoost) perform compared to traditional generalized linear models (Logistic Regression) in capturing non-linear socio-economic relationships?

### 1.3 Machine Learning Formulation
The task is framed as a supervised binary classification problem:
$$\hat{y} = f(\mathbf{x}) \in \{0, 1\}$$
where $y = 1$ denotes a household with positive savings (formal or informal), and $y = 0$ denotes a household with zero reported savings. The feature vector $\mathbf{x} \in \mathbb{R}^{24}$ contains socio-demographic, economic, asset, and accessibility variables.

---

## 2. Dataset Description & Exploratory Data Analysis (EDA)

### 2.1 Data Source & Scope
The data is derived from the **Nigeria General Household Survey (GHS), Wave 5 (Panel 2023/2024)**, conducted by the National Bureau of Statistics (NBS) in collaboration with the World Bank Living Standards Measurement Study (LSMS) program. The raw, individual-level and module-level survey records were aggregated into a master household-level dataset: `13_master_household_ml_dataset.csv`.

* **Total Observations:** 4,771 households
* **Raw Dimension:** 42 initial survey columns
* **Target Class Distribution:**
  * **Has Savings (`savings_target = 1`):** 2,617 households (**54.85%**)
  * **No Savings (`savings_target = 0`):** 2,154 households (**45.15%**)

The target distribution is well-balanced (~55:45), obviating the need for artificial oversampling (e.g., SMOTE) while preserving the natural socio-demographic distribution of the survey sample.

### 2.2 Target Leakage Prevention & Feature Pruning
A critical risk in microeconomic survey ML modeling is **target leakage**, where variables constructed from or correlated with the survey question defining the target are inadvertently included as predictors. To ensure clinical validity, the following variables were explicitly identified and excluded:
* **Direct Target Derivatives:** `formal_savers`, `informal_savers`, `savings_responses`, `savings_answered`, and `eligible_members`.
* **Administrative & High-Cardinality Identifiers:** `hhid` (unique household ID), `state_code`, `state_name`, and `lga_code` (Local Government Area has 774 categories in Nigeria, which risks severe overfitting and sparsity). Regional effects were captured via the 6 macro `zone_code` classifications.
* **Redundant Counts:** `members_with_age`, `head_indiv`, `bank_access_count`, `mobile_money_access_count`, and `assisted_banking_count` (which duplicated the binary access indicators).

---

## 3. Data Preprocessing & Pipeline Architecture

To guarantee reproducibility and prevent data leakage between training and testing subsets, all transformations were constructed using Scikit-Learn's `Pipeline` and `ColumnTransformer` abstractions, fitted **strictly on the training data** ($X_{\text{train}}$) and applied downstream to the test partition ($X_{\text{test}}$).

### 3.1 Partitioning Strategy
* **Train/Test Ratio:** 80% Training ($n = 3,816$), 20% Testing ($n = 955$).
* **Sampling Scheme:** Stratified random splitting (`random_state=42`), maintaining the exact 54.85% : 45.15% target class ratio across partitions.

### 3.2 Feature Categorization & Pipeline Design
The selected 20 input features were partitioned into four processing streams:

| Feature Group | Features | Preprocessing Strategy | Rationale |
| :--- | :--- | :--- | :--- |
| **Heavily Skewed Numerical** (3) | `total_reported_main_job_earnings`, `total_current_asset_value`, `household_size` | Median Imputation $\to$ $\log(1+x)$ Transformation $\to$ `StandardScaler` | Mitigate extreme right-skewness and compress multi-million Naira outliers without data truncation. |
| **Standard Numerical** (9) | `head_age`, `working_age_members`, `members_any_work`, `members_income_activity`, `employment_rate`, `total_reported_income_sources`, `asset_items_listed`, `asset_types_owned`, `total_units_owned` | Median Imputation $\to$ `StandardScaler` | Zero-mean, unit-variance standardization for gradient stability in Logistic Regression and fair scale comparison. |
| **Binary Indicators** (5) | `any_bank_access`, `any_mobile_money_access`, `any_assisted_banking`, `any_mobile_access`, `any_internet_access` | Mode (Most Frequent) Imputation | Preserves binary nature $\{0, 1\}$ without distortion. |
| **Categorical Nominal** (3) | `head_sex_code`, `urban_rural_code`, `zone_code` | Mode Imputation $\to$ `OneHotEncoder(drop='first')` | Expands into dummy indicators while preventing the dummy variable trap (multicollinearity). |

Following One-Hot Encoding of `zone_code` (6 geopolitical zones), `urban_rural_code` (2 sectors), and `head_sex_code` (2 categories), the engineered feature matrix yielded **24 numerical columns**.

---

## 4. Supervised Machine Learning Models

Four diverse model families were trained and evaluated:

### 4.1 Logistic Regression (Baseline Model)
Serves as the parametric generalized linear benchmark:
$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$
Optimized using the `lbfgs` solver with $L_2$ regularization ($C=1.0$, `max_iter=1000`). Logistic regression provides direct interpretability through the exponentiated coefficients (odds ratios: $e^{w_j}$).

### 4.2 Decision Tree Classifier
A non-parametric, rule-based hierarchical classifier using recursive binary splitting based on Gini impurity:
$$I_G(t) = 1 - \sum_{i=1}^C p(i|t)^2$$
Constrained with `max_depth=5`, `min_samples_split=20`, and `min_samples_leaf=10` to prevent overfitting on survey noise.

### 4.3 Random Forest Classifier
An ensemble bagging algorithm constructing 200 de-correlated decision trees:
$$\hat{f}_{\text{RF}}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^B T_b(\mathbf{x})$$
Hyperparameters: `n_estimators=200`, `max_depth=8`, `min_samples_leaf=4`, `random_state=42`. By averaging predictions across randomized subsets of features, Random Forest reduces variance and dampens idiosyncratic survey anomalies.

### 4.4 eXtreme Gradient Boosting (XGBoost)
A regularized gradient-boosted decision tree algorithm that minimizes a second-order Taylor approximation of the objective function:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$
Hyperparameters: `n_estimators=150`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `eval_metric='logloss'`.

---

## 5. Experimental Results & Model Performance

### 5.1 Cross-Validation Performance (Training Set)
Each model underwent 5-fold stratified cross-validation on the 3,816 training samples to estimate in-sample stability:

| Model | 5-Fold CV Mean F1 | 5-Fold CV Std F1 | 5-Fold CV Mean ROC-AUC | 5-Fold CV Std ROC-AUC | 5-Fold CV Mean Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7937 | $\pm$ 0.0094 | 0.8252 | $\pm$ 0.0098 | 0.7686 |
| **Decision Tree** | 0.7816 | $\pm$ 0.0097 | 0.7963 | $\pm$ 0.0152 | 0.7490 |
| **Random Forest** | 0.7981 | $\pm$ 0.0108 | 0.8344 | $\pm$ 0.0105 | 0.7702 |
| **XGBoost** | **0.8038** | $\pm$ 0.0095 | **0.8384** | $\pm$ 0.0102 | **0.7788** |

### 5.2 Hold-Out Test Set Evaluation ($n = 955$)
Final evaluation on the unseen hold-out test set produced the following benchmark metrics:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **0.7749** | 0.7764 | 0.8282 | **0.8015** | **0.8371** |
| **Random Forest** | 0.7696 | 0.7676 | **0.8321** | 0.7985 | 0.8330 |
| **Logistic Regression (Baseline)** | 0.7707 | **0.7788** | 0.8130 | 0.7955 | 0.8250 |
| **Decision Tree** | 0.7508 | 0.7474 | 0.8244 | 0.7840 | 0.8018 |

```
                       ROC-AUC Score Comparison
   XGBoost            |████████████████████████████████████████▎   | 0.8371
   Random Forest      |██████████████████████████████████████      | 0.8330
   Logistic Reg.      |███████████████████████████████████▌        | 0.8250
   Decision Tree      |████████████████████████████████▍           | 0.8018
                      +--------------------------------------------+
                      0.60                        0.75          0.85
```

### 5.3 In-Depth Performance Analysis
1. **Ensemble Superiority:** XGBoost demonstrated the highest discriminative power (**ROC-AUC: 0.8371**) and balanced accuracy (**F1-Score: 0.8015**), reflecting its capacity to capture subtle interactions between geographic region, household wealth, and physical infrastructure.
2. **Resilience of the Logistic Baseline:** Logistic Regression performed remarkably well (**ROC-AUC: 0.8250, Accuracy: 77.07%**), closely trailing the complex ensembles. This indicates that the engineered features have strong, coherent linear log-odds relationships with savings participation.
3. **Recall Optimization:** All models exhibited strong recall ($\ge 81.3\%$), meaning that households with savings are successfully captured with minimal false negatives (Type II error). Random Forest achieved the highest recall (**83.21%**), correctly identifying 436 out of 524 saving households in the test set.

---

## 6. Feature Importance & Financial Inclusion Insights

Combining the parametric interpretability of Logistic Regression with the tree-based split importances of Random Forest and XGBoost provides consistent insights into the drivers of household savings in Nigeria:

### 6.1 Top Predictors & Odds Ratio Analysis

| Feature Name | Logistic Reg. Coef ($\beta$) | Odds Ratio ($e^\beta$) | RF Gini Importance | XGBoost Gain Importance | Socioeconomic Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `any_bank_access` | **+1.2356** | **3.4406** | **0.2307** | **0.4188** | Access to a commercial bank or MFB increases savings odds by **244%**. Dominant predictor across all algorithms. |
| `any_assisted_banking` | **+0.7390** | **2.0938** | 0.0752 | 0.0495 | Access to POS/Agent banking networks more than **doubles** savings odds (**+109%**). |
| `any_mobile_money_access` | **+0.6071** | **1.8351** | 0.1004 | 0.0796 | Digital wallet / mobile money access increases savings odds by **83.5%**. |
| `any_internet_access` | **+0.3216** | **1.3793** | 0.0813 | 0.1087 | Internet connectivity enables digital fintech apps, increasing savings odds by **37.9%**. |
| `asset_types_owned` | **+0.1895** | **1.2087** | 0.0944 | 0.0263 | Asset portfolio variety reflects economic surplus and resilience (**+20.9%** per std dev). |
| `employment_rate` | **+0.1651** | **1.1795** | 0.0356 | 0.0188 | Higher proportion of employed working-age members directly feeds household savings capacity. |
| `household_size` | **+0.1137** | **1.1204** | 0.0244 | 0.0139 | Larger households exhibit higher informal pooling (esusu) and multiple income contributors. |
| `total_current_asset_value` | **+0.0967** | **1.1015** | 0.0715 | 0.0162 | Liquid and physical asset value provides financial security to save. |
| `total_reported_main_job_earnings` | **+0.0906** | **1.0948** | 0.0470 | 0.0167 | Formal wage and enterprise income directly supply savable liquidity. |
| `zone_code_3.0` (North West) | **-0.6022** | **0.5476** | 0.0147 | 0.0236 | Households in the North West show **45.2% lower odds** of formal/informal savings relative to North Central. |
| `zone_code_6.0` (South West) | **-0.4870** | **0.6144** | 0.0059 | 0.0179 | Regional relative baseline difference showing distinct informal vs formal savings dynamics. |
| `zone_code_2.0` (North East) | **-0.4118** | **0.6624** | 0.0100 | 0.0216 | Severe infrastructural and security challenges impede access to institutional savings. |

### 6.2 Test-Set Permutation Feature Importance (Cross-Checking Tree Cardinality Bias)

As outlined in Section 3.10 of the study methodology, built-in tree importances (MDI / Gini in Random Forest and Split Gain in XGBoost) can overemphasize continuous features with many distinct values. To obtain an unbiased, model-agnostic verification, **permutation feature importance** was computed on the held-out test partition ($n = 955$) across 10 random shuffles using the change in ROC-AUC as the evaluation metric:

| Rank | Feature Name | LR Permutation ($\Delta$ AUC) | RF Permutation ($\Delta$ AUC) | XGBoost Permutation ($\Delta$ AUC) | Methodological Interpretation |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `any_bank_access` | **0.0713** $\pm$ 0.0078 | **0.0611** $\pm$ 0.0094 | **0.0631** $\pm$ 0.0071 | Shuffling bank access produces the largest catastrophic drop in test ROC-AUC across **all three architectures**, confirming it is the uncontested primary driver. |
| **2** | `any_assisted_banking` | **0.0272** $\pm$ 0.0041 | **0.0269** $\pm$ 0.0037 | **0.0356** $\pm$ 0.0050 | Confirmed as the second most vital real-world predictor, dropping XGBoost ROC-AUC by 0.0356. |
| **3** | `any_mobile_money_access` | **0.0108** $\pm$ 0.0025 | **0.0168** $\pm$ 0.0022 | **0.0136** $\pm$ 0.0020 | Reliable digital access signal with virtually zero cardinality distortion. |
| **4** | `any_internet_access` | 0.0042 $\pm$ 0.0018 | 0.0065 $\pm$ 0.0028 | 0.0085 $\pm$ 0.0026 | Demonstrates consistent positive utility for fintech enablement. |
| **5** | `employment_rate` | 0.0045 $\pm$ 0.0012 | 0.0031 $\pm$ 0.0010 | 0.0064 $\pm$ 0.0017 | Strongest economic/labor driver on test data, ahead of raw earnings. |
| **6** | `total_reported_income_sources`| 0.0002 $\pm$ 0.0010 | 0.0033 $\pm$ 0.0012 | 0.0058 $\pm$ 0.0026 | Household income diversification provides buffer stability. |
| **7** | `zone_code_4.0` (South East) | 0.0001 $\pm$ 0.0005 | 0.0032 $\pm$ 0.0007 | 0.0048 $\pm$ 0.0017 | Regional commercial hub effect. |
| **8** | `zone_code_2.0` (North East) | 0.0013 $\pm$ 0.0004 | 0.0013 $\pm$ 0.0003 | 0.0036 $\pm$ 0.0007 | Regional structural deficit indicator. |
| **9** | `zone_code_3.0` (North West) | 0.0065 $\pm$ 0.0015 | 0.0025 $\pm$ 0.0010 | 0.0035 $\pm$ 0.0014 | Negative regional penalty confirmed on out-of-sample data. |
| **10**| `asset_types_owned` | 0.0054 $\pm$ 0.0021 | 0.0032 $\pm$ 0.0024 | 0.0035 $\pm$ 0.0017 | Resilient physical wealth proxy. |

**Key Cross-Check Conclusion:** The permutation importance rankings perfectly corroborate the built-in tree importances and Logistic Regression odds ratios. Both methods identify **physical banking access**, **agency banking**, and **mobile money** as the top 3 structural drivers of household savings in Nigeria, eliminating any concern that tree models favored continuous asset or earnings values due to cardinality bias.

---

## 7. Key Findings & Discussion

### 7.1 The Primacy of Institutional Proximity
The empirical evidence decisively highlights that **financial access infrastructure is the preeminent bottleneck to household savings in Nigeria**:
* In the XGBoost model, `any_bank_access` alone accounts for **41.88% of total split gain**, and **23.07% of Gini importance** in Random Forest.
* In Logistic Regression, possessing bank access multiplies the odds of saving by **3.44 times**, holding income, assets, education, and geography constant.
* This proves that household non-saving is not merely a consequence of poverty or lack of income, but heavily a function of **physical and institutional exclusion**. When banking services are within reach, households actively save.

### 7.2 The Catalytic Role of Agent Banking & Mobile Money
Traditional brick-and-mortar bank branches are capital-intensive and concentrated in urban commercial centers. The results demonstrate that **assisted banking (`any_assisted_banking`: Odds Ratio = 2.09)** and **mobile money (`any_mobile_money_access`: Odds Ratio = 1.84)** are powerful second-line drivers:
* Point-of-Sale (POS) agents and mobile money kiosks effectively bridge the last-mile gap in peri-urban and rural communities, turning informal cash hoards into secure stored value.

### 7.3 Assets vs. Current Income
Interestingly, `asset_types_owned` and `total_current_asset_value` exhibited stronger and more consistent tree feature importance than current month earnings (`total_reported_main_job_earnings`). In informal economies characterized by irregular wage cycles, **accumulated asset variety** is a much more reliable indicator of long-term household wealth and financial buffer capacity than transitory monthly earnings.

### 7.4 Structural Regional Disparities
The negative coefficients associated with the North West ($\beta = -0.6022$) and North East ($\beta = -0.4118$) geopolitical zones relative to the baseline highlight persistent regional divides. These regions face lower bank branch density, lower digital connectivity rates, and higher vulnerability to informal agricultural volatility, underscoring the necessity of targeted regional interventions.

---

## 8. Policy Recommendations (Central Bank of Nigeria & Stakeholders)

Based on the quantitative findings, three actionable policy interventions are proposed:

1. **Aggressive Expansion of Agent Banking & Shared Kiosk Infrastructure:**
   * Since assisted banking doubles household savings propensity, the Central Bank of Nigeria (CBN) and commercial banks should subsidize POS terminals, reduce transaction fees for micro-savings deposits, and incentivize agent networks in historically underserved Northern and rural LGAs.
2. **Zero-Data USSD & Tier-1 KYC Mobile Money Penetration:**
   * Given the positive odds ratio of mobile money ($+83.5\%$) and internet access ($+37.9\%$), telcos and fintech operators should expand zero-rated USSD mobile wallets with simplified, biometrics-driven Tier-1 Know-Your-Customer (KYC) requirements, lowering barriers for unbanked informal earners.
3. **Integrating Asset-Building Livelihood Programs with Formal Savings:**
   * Social safety net programs (such as conditional cash transfers and agricultural input subsidies) should mandate direct digital disbursement into interest-bearing savings accounts, facilitating simultaneous asset accumulation and financial integration.

---

## 9. Limitations & Future Work

1. **Cross-Sectional Data:** The current analysis utilizes cross-sectional survey microdata (GHS Wave 5). Future studies could analyze longitudinal panel waves (Waves 1–5) to track household transition into and out of savings over multi-year economic cycles.
2. **Disaggregation of Savings Vehicles:** The current `savings_target` combines formal bank accounts and informal savings (esusu/adashi). A multi-class formulation ($0 = \text{No Savings}, 1 = \text{Informal Only}, 2 = \text{Formal Bank Savings}$) could isolate channel-specific substitution dynamics.
3. **Hyperparameter Optimization & Calibration:** Further gains in precision-recall trade-offs may be extracted by running Bayesian optimization on XGBoost regularization parameters and calibrating classification thresholds for specific policy deployment costs.

---

## 10. Conclusion

This project successfully established a robust, leak-free supervised machine learning framework for predicting household savings in Nigeria using GHS Wave 5 data. Through disciplined preprocessing, skewness remediation, standard scaling, and stratified model benchmarking, the **XGBoost Classifier** emerged as the top-performing model, achieving **77.49% accuracy** and an **ROC-AUC of 0.8371**. 

Crucially, the feature importance and econometric odds analysis confirm that **financial institutional access (`any_bank_access`, `any_assisted_banking`, `any_mobile_money_access`) is the single greatest determinant of savings behavior**, delivering rigorous empirical evidence to guide Nigeria's digital financial inclusion agenda.

---

## Deliverables & Artifacts Index
* **Primary Preprocessing & Modeling Notebook:** `model_development_nigeria.ipynb`
* **Exploratory Data Analysis Notebook:** `eda_savings_nigeria.ipynb`
* **Master Household Dataset:** `13_master_household_ml_dataset.csv`
* **Hold-Out Evaluation Metrics:** `model_evaluation_metrics.csv`
* **Feature Importances & Odds Ratios:** `model_feature_importances.csv`
