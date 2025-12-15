# Interim Report: Credit Risk Model Implementation

## 1. Understanding and Defining the Business Objective
**Objective:** The primary goal is to develop a credit scoring model for Bati Bank's "Buy-Now-Pay-Later" service. This involves transforming raw eCommerce transaction data into a predictive risk metric to facilitate automated loan approvals.

**Business Context & Regulatory Framework:**
*   **Basel II Accord**: The model must comply with Basel II capital requirements. This necessitates a robust estimation of risk parameters (Probability of Default) and, crucially, **interpretability**. The model cannot be a "black box" regarding capital allocation decisions; every score must be explainable to regulators and auditors.
*   **The Proxy Variable Strategy**: Unlike traditional banking reference data, we lack a labeled "default" dataset. We are therefore defining a **proxy variable** for creditworthiness based on **RFM (Recency, Frequency, Monetary)** behavior.
    *   *Premise*: Users with consistent, high-value, and recent transaction history are lower risk.
    *   *Risk*: The risk of misclassification bias is non-trivial. A low-frequency user is not necessarily a "bad" borrower, just an inactive one. The model must carefully distinguish between "churned/inactive" and "risky."
    *   *Target Variable Engineering*: We will use **K-Means clustering** on RFM features to segment customers into risk categories (e.g., high-risk, medium-risk, low-risk), which will serve as our proxy target variable for credit risk prediction.

## 2. Discussion of Completed Work and Initial Analysis (Task 1 & 2)
We have successfully set up the project structure and performed a comprehensive Exploratory Data Analysis (EDA) on the provided dataset of 95,662 transactions.

### 2.1 Dataset Structure and Overview

**Dataset Dimensions:**
*   **Total Records**: 95,662 transactions
*   **Total Features**: 16 columns
*   **Data Types**: 11 object (categorical) columns, 4 integer columns, 1 float column
*   **Memory Usage**: ~11.7 MB

**Feature Categories:**
*   **Identifiers**: TransactionId, BatchId, AccountId, SubscriptionId, CustomerId
*   **Geographic**: CurrencyCode, CountryCode
*   **Transaction Attributes**: ProviderId, ProductId, ProductCategory, ChannelId
*   **Financial**: Amount (signed transaction value), Value (absolute transaction value)
*   **Temporal**: TransactionStartTime
*   **Business Logic**: PricingStrategy
*   **Note**: `FraudResult` exists in the dataset but is **not** our target variable. It represents fraud detection labels, which is distinct from credit risk assessment.

*Visualization Reference: See EDA notebook Cell 3-4 for dataset structure and data types summary.*

### 2.2 Data Quality Assessment

**Missing Values Analysis:**
*   **Result**: Zero missing values across all 16 columns
*   **Implication**: No imputation required, simplifying the preprocessing pipeline

*Visualization Reference: See EDA notebook Cell 9 for missing values heatmap (shows no missing data).*

### 2.3 Summary Statistics

**Numeric Features Summary:**
*   **Amount**: Can be positive (debits) or negative (credits), representing bidirectional cash flow
*   **Value**: Absolute transaction value, showing the magnitude of transactions
*   **CountryCode, PricingStrategy**: Categorical numeric codes requiring encoding

**Categorical Features Summary:**
*   **High Cardinality Features**: ProviderId, ProductId, ChannelId show significant variation
*   **Customer-Level Aggregation Required**: Multiple transactions per CustomerId, necessitating aggregation for customer-level modeling

*Visualization Reference: See EDA notebook Cell 6-7 for detailed summary statistics tables.*

### 2.4 Distribution Analysis

**Transaction Amount Distributions:**
*   **Amount and Value**: Both show right-skewed distributions with potential outliers
*   **Implication**: Normalization/standardization will be required for model training, particularly for distance-based algorithms

*Visualization Reference: See EDA notebook Cell 11 for distribution histograms of Amount and Value.*

**Note on FraudResult Distribution:**
*   The `FraudResult` variable shows extreme class imbalance (~0.2% positive class), but this is **not relevant** for our credit risk model. Our target variable will be derived from RFM-based customer segmentation using K-Means clustering.

*Visualization Reference: See EDA notebook Cell 12 for FraudResult class distribution (shown for completeness, not as target).*

### 2.5 Correlation Analysis

**Key Correlations Identified:**
*   **Value and Amount**: Strong positive correlation (expected, as Value = |Amount|)
*   **Numeric Features**: Moderate correlations between financial features
*   **Implication**: Feature selection and dimensionality reduction may be beneficial to avoid multicollinearity

*Visualization Reference: See EDA notebook Cell 14 for correlation heatmap matrix showing all numeric feature relationships.*

### 2.6 Outlier Detection

**Outlier Patterns:**
*   **Value Feature**: Significant outliers present, particularly in certain transaction categories
*   **Implication**: Robust scaling methods (e.g., RobustScaler) may be preferred over StandardScaler to handle outliers during normalization

*Visualization Reference: See EDA notebook Cell 16 for boxplot analysis of Value by transaction category.*

### 2.7 Key Insights Summary

1.   **Data Completeness**: 100% complete dataset with zero missing values—no imputation pipeline needed.
2.   **Feature Engineering Requirements**: 
    *   Customer-level aggregation required (transaction → customer transformation)
    *   Temporal feature extraction needed from `TransactionStartTime` (e.g., day of week, month, time since last transaction)
    *   Categorical encoding required for high-cardinality features
3.   **Preprocessing Needs**: 
    *   Normalization/standardization required for numeric features (consider RobustScaler for outlier robustness)
    *   Feature selection to address multicollinearity
4.   **Target Variable**: Will be engineered using **K-Means clustering** on RFM features, not using `FraudResult`

## 3. Next Steps and Key Areas of Focus

### 3.1 Immediate Focus (Task 3 - Feature Engineering)

**RFM Feature Construction:**
*   Aggregate transaction-level data to customer level
*   Calculate **Recency**: Days since most recent transaction
*   Calculate **Frequency**: Total number of transactions per customer
*   Calculate **Monetary**: Total transaction value, average transaction value, and transaction value variability
*   These RFM features will form the basis of our credit risk proxy

**Proxy Target Variable Engineering:**
*   Apply **K-Means clustering** on RFM features to segment customers into risk categories
*   We start with **k=3** clusters to mirror credit-grade tiers (high/medium/low risk). When a cohort has fewer than three customers, we automatically reduce *k* to the number of available samples and mark the lowest TotalTransactionAmount cluster as high risk. This keeps the proxy target stable even for very small cohorts.
*   The high-risk label is explicitly defined as the cluster with the **lowest TotalTransactionAmount** and **frequency**, which captures disengaged or low-value users.
*   Each clustering run logs its inertia and centers (via the debug instrumentation and MLflow runs) to provide diagnostics for stakeholders.
*   This creates our binary/multi-class proxy target variable for credit risk

**Temporal Feature Extraction:**
*   Extract features from `TransactionStartTime`:
    *   Day of week, month, hour
    *   Time since first transaction (customer tenure)
    *   Time between transactions (transaction intervals)
    *   Seasonal patterns (holiday periods, etc.)

**WoE and IV Implementation:**
*   Implement Weight of Evidence (WoE) binning to transform features into a linear scale
*   Calculate Information Value (IV) for feature selection
*   Suitable for scorecard development and improved interpretability

**Normalization and Standardization:**
*   Apply RobustScaler or StandardScaler to numeric features
*   Decision will be based on outlier sensitivity analysis
*   Critical for distance-based algorithms and gradient-based optimization

### 3.2 Future Focus (Task 4 & 5 - Modeling & MLOps)

**Model Training Strategy:**
*   **Baseline Models**: Train Logistic Regression (interpretable) vs. XGBoost (complex, high performance)
*   **Hyperparameter Tuning Strategy**: 
    *   Use GridSearchCV or RandomizedSearchCV for systematic hyperparameter optimization
    *   Focus on regularization parameters (C for Logistic Regression, alpha/lambda for XGBoost)
    *   Optimize class weights to handle potential imbalance in proxy target variable
    *   Cross-validation with temporal splits to prevent data leakage
*   **Model Selection**: Evaluate trade-off between pure performance (XGBoost) and interpretability (Logistic Regression) for Basel II compliance

**Deployment Infrastructure:**
*   **API Development**: FastAPI endpoint for real-time predictions
*   **Containerization**: Docker for consistent deployment environments
*   **Model Tracking**: MLflow for experiment tracking, model versioning, and reproducibility
*   **Monitoring**: Log prediction distributions and model performance metrics

## 4. Report Structure and Clarity
This report follows a structured approach to document the lifecycle of the credit risk project, ensuring that technical findings are always tied back to the business objective of enabling sustainable credit products for Bati Bank. All EDA findings are supported by visualizations and tables available in the `notebooks/eda.ipynb` notebook, with specific cell references provided for transparency and reproducibility.
