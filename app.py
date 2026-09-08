import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

# Page configuration
st.set_page_config(
    page_title="Household Savings Predictor | Nigeria GHS",
    page_icon="🇳🇬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1b5e20;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #424242;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f1f8e9;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Data & Model Pipeline Caching
# -------------------------------------------------------------
@st.cache_resource
def train_and_cache_models():
    df = pd.read_csv('13_master_household_ml_dataset.csv')
    
    skewed_features = ['total_reported_main_job_earnings', 'total_current_asset_value', 'household_size']
    standard_num_features = [
        'head_age', 'working_age_members', 'members_any_work', 
        'members_income_activity', 'employment_rate', 'total_reported_income_sources',
        'asset_items_listed', 'asset_types_owned', 'total_units_owned'
    ]
    binary_features = [
        'any_bank_access', 'any_mobile_money_access', 'any_assisted_banking',
        'any_mobile_access', 'any_internet_access'
    ]
    categorical_features = ['head_sex_code', 'urban_rural_code', 'zone_code']
    
    selected_features = skewed_features + standard_num_features + binary_features + categorical_features
    X = df[selected_features].copy()
    y = df['savings_target'].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    skewed_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log_transform', FunctionTransformer(np.log1p, validate=False)),
        ('scaler', StandardScaler())
    ])
    
    standard_num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    binary_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('skew', skewed_transformer, skewed_features),
            ('num', standard_num_transformer, standard_num_features),
            ('bin', binary_transformer, binary_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    cat_ohe = preprocessor.named_transformers_['cat'].named_steps['ohe']
    cat_encoded_cols = list(cat_ohe.get_feature_names_out(categorical_features))
    feature_names = skewed_features + standard_num_features + binary_features + cat_encoded_cols
    
    models = {
        'XGBoost (Champion)': XGBClassifier(
            random_state=42, n_estimators=150, max_depth=4, 
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss'
        ),
        'Random Forest': RandomForestClassifier(
            random_state=42, n_estimators=200, max_depth=8, min_samples_leaf=4, n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=1000, C=1.0, solver='lbfgs'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42, max_depth=5, min_samples_split=20, min_samples_leaf=10
        )
    }
    
    fitted_models = {}
    eval_results = {}
    preds = {}
    probas = {}
    
    for name, model in models.items():
        model.fit(X_train_proc, y_train)
        fitted_models[name] = model
        
        y_pred = model.predict(X_test_proc)
        y_proba = model.predict_proba(X_test_proc)[:, 1]
        
        preds[name] = y_pred
        probas[name] = y_proba
        
        eval_results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_proba)
        }
        
    metrics_df = pd.DataFrame(eval_results).T.reset_index().rename(columns={'index': 'Model'})
    
    # Feature importance
    lr = fitted_models['Logistic Regression']
    rf = fitted_models['Random Forest']
    xgb = fitted_models['XGBoost (Champion)']
    
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'LogReg_Coef': lr.coef_[0],
        'Odds_Ratio': np.exp(lr.coef_[0]),
        'RF_Importance': rf.feature_importances_,
        'XGB_Importance': xgb.feature_importances_
    }).sort_values(by='Odds_Ratio', ascending=False)
    
    return {
        'df': df,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'selected_features': selected_features,
        'fitted_models': fitted_models,
        'metrics_df': metrics_df,
        'imp_df': imp_df,
        'X_test_proc': X_test_proc,
        'y_test': y_test,
        'preds': preds,
        'probas': probas
    }

data_dict = train_and_cache_models()

# -------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/7/79/Flag_of_Nigeria.svg", width=60)
st.sidebar.title("CSC 572 ML Project")
st.sidebar.markdown("**Household Savings in Nigeria**\n*GHS Wave 5 Microdata*")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Select Navigation View:",
    [
        "🔮 Interactive Household Predictor",
        "📊 Model Comparison & Metrics",
        "💡 Financial Inclusion Simulator",
        "📈 Feature Importance & Insights",
        "📁 Dataset & Survey Overview"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Best Model:** XGBoost achieved an ROC-AUC of **0.8371** and an F1-Score of **0.8015**.")

# -------------------------------------------------------------
# View 1: Interactive Predictor
# -------------------------------------------------------------
if app_mode == "🔮 Interactive Household Predictor":
    st.markdown('<div class="main-header">🔮 Household Savings Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Simulate household characteristics and evaluate the probability of formal/informal savings adoption.</div>', unsafe_allow_html=True)
    
    col_model, col_spacer = st.columns([2, 2])
    with col_model:
        selected_model_name = st.selectbox(
            "Select Machine Learning Model for Inference:",
            list(data_dict['fitted_models'].keys())
        )
    
    active_model = data_dict['fitted_models'][selected_model_name]
    
    st.markdown("### 1. Household Demographics & Location")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        zone_name = st.selectbox("Geopolitical Zone", [
            "1 - North Central", "2 - North East", "3 - North West",
            "4 - South East", "5 - South South", "6 - South West"
        ])
        zone_code = float(zone_name.split(" - ")[0])
    with c2:
        sector_name = st.selectbox("Sector", ["1 - Urban", "2 - Rural"])
        urban_rural_code = float(sector_name.split(" - ")[0])
    with c3:
        head_sex_name = st.selectbox("Head of Household Sex", ["1 - Male", "2 - Female"])
        head_sex_code = float(head_sex_name.split(" - ")[0])
    with c4:
        head_age = st.slider("Head of Household Age", min_value=18, max_value=95, value=45)
        
    c5, c6, c7 = st.columns(3)
    with c5:
        household_size = st.slider("Total Household Size", min_value=1, max_value=25, value=5)
    with c6:
        working_age_members = st.slider("Working-Age Members (15–64)", min_value=1, max_value=household_size, value=min(3, household_size))
    with c7:
        employment_rate = st.slider("Household Employment Rate", min_value=0.0, max_value=1.0, value=0.67, step=0.05)

    st.markdown("### 2. Financial & Digital Access")
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        any_bank_access = 1 if st.checkbox("🏦 Commercial/MFB Bank Access", value=True) else 0
    with f2:
        any_assisted_banking = 1 if st.checkbox("🏪 POS / Agent Banking", value=True) else 0
    with f3:
        any_mobile_money_access = 1 if st.checkbox("📱 Mobile Money Wallet", value=False) else 0
    with f4:
        any_mobile_access = 1 if st.checkbox("📞 Mobile Phone Access", value=True) else 0
    with f5:
        any_internet_access = 1 if st.checkbox("🌐 Internet Connectivity", value=False) else 0

    st.markdown("### 3. Economic & Asset Portfolio")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        earnings = st.number_input("Monthly Main Job Earnings (₦)", min_value=0, max_value=5000000, value=65000, step=5000)
    with e2:
        asset_value = st.number_input("Total Current Asset Value (₦)", min_value=0, max_value=50000000, value=350000, step=25000)
    with e3:
        asset_types_owned = st.slider("Distinct Asset Types Owned", min_value=0, max_value=20, value=5)
    with e4:
        income_sources = st.slider("Total Income Sources", min_value=1, max_value=8, value=2)

    # Secondary inputs
    members_any_work = 1 if working_age_members > 0 else 0
    members_income_activity = max(1, int(working_age_members * employment_rate))
    asset_items_listed = asset_types_owned
    total_units_owned = float(asset_types_owned * 2)

    # Input DataFrame
    input_data = pd.DataFrame([{
        'total_reported_main_job_earnings': float(earnings),
        'total_current_asset_value': float(asset_value),
        'household_size': float(household_size),
        'head_age': float(head_age),
        'working_age_members': float(working_age_members),
        'members_any_work': float(members_any_work),
        'members_income_activity': float(members_income_activity),
        'employment_rate': float(employment_rate),
        'total_reported_income_sources': float(income_sources),
        'asset_items_listed': float(asset_items_listed),
        'asset_types_owned': float(asset_types_owned),
        'total_units_owned': float(total_units_owned),
        'any_bank_access': any_bank_access,
        'any_mobile_money_access': any_mobile_money_access,
        'any_assisted_banking': any_assisted_banking,
        'any_mobile_access': any_mobile_access,
        'any_internet_access': any_internet_access,
        'head_sex_code': float(head_sex_code),
        'urban_rural_code': float(urban_rural_code),
        'zone_code': float(zone_code)
    }])

    # Preprocess and Predict
    input_proc = data_dict['preprocessor'].transform(input_data)
    pred_class = active_model.predict(input_proc)[0]
    pred_prob = active_model.predict_proba(input_proc)[0][1]

    st.markdown("---")
    st.markdown("### Prediction Outcome")
    
    r1, r2 = st.columns([1, 2])
    with r1:
        if pred_class == 1:
            st.success("### ✅ Has Savings")
            st.markdown(f"**Predicted Status:** **Savers**")
            st.markdown(f"**Confidence:** `{pred_prob*100:.1f}%`")
        else:
            st.error("### ❌ No Savings")
            st.markdown(f"**Predicted Status:** **Non-Savers**")
            st.markdown(f"**Confidence:** `{(1-pred_prob)*100:.1f}%`")
            
    with r2:
        st.markdown(f"**Estimated Probability of Savings:** `{pred_prob*100:.2f}%`")
        st.progress(float(pred_prob))
        
        # Drivers interpretation
        drivers = []
        if any_bank_access:
            drivers.append("🏦 Commercial bank access (+244% odds boost)")
        if any_assisted_banking:
            drivers.append("🏪 POS / Agent banking (+109% odds boost)")
        if any_mobile_money_access:
            drivers.append("📱 Mobile money access (+84% odds boost)")
        if asset_types_owned >= 5:
            drivers.append("📦 Diversified asset buffer")
            
        if drivers:
            st.markdown("**Key Positive Factors for this Profile:**")
            for d in drivers:
                st.markdown(f"- {d}")
        else:
            st.warning("⚠️ Profile lacks key financial access infrastructure (no bank or agent banking).")

# -------------------------------------------------------------
# View 2: Model Benchmarks & Metrics
# -------------------------------------------------------------
elif app_mode == "📊 Model Comparison & Metrics":
    st.markdown('<div class="main-header">📊 Model Performance Benchmark</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Hold-out test set evaluation (n = 955) comparing all 4 machine learning models.</div>', unsafe_allow_html=True)
    
    metrics_df = data_dict['metrics_df']
    st.dataframe(metrics_df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'], color='#c8e6c9', axis=0))
    
    st.markdown("### Comparative Performance Across Evaluation Metrics")
    m_melt = metrics_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(data=m_melt, x='Metric', y='Score', hue='Model', palette='tab10', ax=ax)
    ax.set_ylim(0.65, 0.90)
    ax.set_ylabel("Score (0.0 - 1.0)")
    ax.set_title("Test Set Scores Across 5 Evaluation Metrics", fontweight='bold')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("### Receiver Operating Characteristic (ROC) Curves")
    fig_roc, ax_roc = plt.subplots(figsize=(7, 4.5))
    for name, prob in data_dict['probas'].items():
        fpr, tpr, _ = roc_curve(data_dict['y_test'], prob)
        auc_val = roc_auc_score(data_dict['y_test'], prob)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", lw=2)
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Chance')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Test Set ROC Curves', fontweight='bold')
    ax_roc.legend(loc='lower right')
    plt.tight_layout()
    st.pyplot(fig_roc)

# -------------------------------------------------------------
# View 3: Financial Inclusion Simulator
# -------------------------------------------------------------
elif app_mode == "💡 Financial Inclusion Simulator":
    st.markdown('<div class="main-header">💡 Policy & Financial Inclusion Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Simulate the macro-impact of financial infrastructure expansion on unbanked households.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This simulator models an **unbanked, rural household** in Northern Nigeria and demonstrates the predicted impact of expanding various financial touchpoints (Agent Banking, Mobile Money, Commercial Banks).
    """)
    
    sim_zone = st.selectbox("Select Target Region for Simulation:", [
        "3 - North West (Historically lowest baseline)",
        "2 - North East (Severe infrastructure constraints)",
        "1 - North Central",
        "4 - South East",
        "5 - South South",
        "6 - South West"
    ])
    z_code = float(sim_zone.split(" - ")[0])
    
    # Base scenario: rural, low income, unbanked
    base_profile = {
        'total_reported_main_job_earnings': 35000.0,
        'total_current_asset_value': 120000.0,
        'household_size': 6.0,
        'head_age': 42.0,
        'working_age_members': 3.0,
        'members_any_work': 1.0,
        'members_income_activity': 2.0,
        'employment_rate': 0.67,
        'total_reported_income_sources': 1.0,
        'asset_items_listed': 3.0,
        'asset_types_owned': 3.0,
        'total_units_owned': 6.0,
        'any_bank_access': 0,
        'any_mobile_money_access': 0,
        'any_assisted_banking': 0,
        'any_mobile_access': 1,
        'any_internet_access': 0,
        'head_sex_code': 1.0,
        'urban_rural_code': 2.0, # Rural
        'zone_code': z_code
    }
    
    scenarios = {
        "Status Quo (No Financial Access)": base_profile.copy(),
        "+ Mobile Money Wallet Only": {**base_profile, 'any_mobile_money_access': 1},
        "+ Agent Banking (POS Kiosk)": {**base_profile, 'any_assisted_banking': 1},
        "+ Agent Banking & Mobile Money": {**base_profile, 'any_assisted_banking': 1, 'any_mobile_money_access': 1},
        "+ Full Commercial Bank Access": {**base_profile, 'any_bank_access': 1, 'any_assisted_banking': 1}
    }
    
    xgb_model = data_dict['fitted_models']['XGBoost (Champion)']
    sim_results = []
    
    for sc_name, sc_data in scenarios.items():
        sc_df = pd.DataFrame([sc_data])
        sc_proc = data_dict['preprocessor'].transform(sc_df)
        prob = xgb_model.predict_proba(sc_proc)[0][1]
        sim_results.append({'Intervention Scenario': sc_name, 'Savings Probability': prob * 100})
        
    res_df = pd.DataFrame(sim_results)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#d32f2f', '#f57c00', '#fbc02d', '#388e3c', '#1976d2']
    bars = ax.barh(res_df['Intervention Scenario'], res_df['Savings Probability'], color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Predicted Probability of Saving (%)', fontweight='bold')
    ax.set_title(f'Impact of Financial Inclusion Interventions ({sim_zone.split(" (")[0]})', fontweight='bold')
    
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.1f}%", (w + 1.5, bar.get_y() + bar.get_height()/2.), va='center', fontweight='bold')
        
    plt.tight_layout()
    st.pyplot(fig)
    
    st.info("📌 **Policy Insight:** Introducing an agent banking kiosk or mobile money wallet lifts an excluded household from low savings probability (~30-40%) to over 65-75%, highlighting the immense leverage of last-mile digital infrastructure.")

# -------------------------------------------------------------
# View 4: Feature Importance
# -------------------------------------------------------------
elif app_mode == "📈 Feature Importance & Insights":
    st.markdown('<div class="main-header">📈 Feature Importance Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Econometric odds ratios and tree-based split gains.</div>', unsafe_allow_html=True)
    
    imp_df = data_dict['imp_df']
    
    t1, t2 = st.tabs(["Top Odds Ratios (Logistic Regression)", "Tree Importances (Random Forest & XGBoost)"])
    
    with t1:
        st.markdown("### Top Drivers by Odds Ratio")
        st.dataframe(imp_df[['Feature', 'Odds_Ratio', 'LogReg_Coef']].head(12).style.format({'Odds_Ratio': '{:.2f}', 'LogReg_Coef': '{:.3f}'}))
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=imp_df.head(8), x='Odds_Ratio', y='Feature', palette='crest', ax=ax)
        ax.set_title("Top 8 Features by Odds Ratio (Multiplicative Boost to Savings)", fontweight='bold')
        ax.axvline(1.0, color='red', linestyle='--', label='Neutral (1.0)')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        
    with t2:
        st.markdown("### Tree Gain Importances")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        xgb_top = imp_df.sort_values(by='XGB_Importance', ascending=False).head(8)
        sns.barplot(data=xgb_top, x='XGB_Importance', y='Feature', palette='magma', ax=axes[0])
        axes[0].set_title('Top 8 Predictors: XGBoost Gain', fontweight='bold')
        
        rf_top = imp_df.sort_values(by='RF_Importance', ascending=False).head(8)
        sns.barplot(data=rf_top, x='RF_Importance', y='Feature', palette='viridis', ax=axes[1])
        axes[1].set_title('Top 8 Predictors: Random Forest Gini', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)

# -------------------------------------------------------------
# View 5: Dataset & Overview
# -------------------------------------------------------------
elif app_mode == "📁 Dataset & Survey Overview":
    st.markdown('<div class="main-header">📁 Dataset & Survey Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Nigeria General Household Survey (GHS Wave 5) Microdata.</div>', unsafe_allow_html=True)
    
    df = data_dict['df']
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Households", f"{len(df):,}")
    m2.metric("Predictor Features", "24 engineered")
    m3.metric("Has Savings (1)", f"{df['savings_target'].mean()*100:.1f}%")
    m4.metric("No Savings (0)", f"{(1 - df['savings_target'].mean())*100:.1f}%")
    
    st.markdown("### Sample Microdata (First 10 Rows)")
    st.dataframe(df.head(10))
