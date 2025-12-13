# Interim Report: Credit Risk Model Implementation

## 1. Understanding and Defining the Business Objective
**Objective:** The primary goal is to develop a credit scoring model for Bati Bank's "Buy-Now-Pay-Later" service. This involves transforming raw eCommerce transaction data into a predictive risk metric to facilitate automated loan approvals.

**Business Context & Regulatory Framework:**
*   **Basel II Accord**: The model must comply with Basel II capital requirements. This necessitates a robust estimation of risk parameters (Probability of Default) and, crucially, **interpretability**. The model cannot be a "black box" regarding capital allocation decisions; every score must be explainable to regulators and auditors.
*   **The Proxy Variable Strategy**: Unlike traditional banking reference data, we lack a labeled "default" dataset. We are therefore defining a **proxy variable** for creditworthiness based on **RFM (Recency, Frequency, Monetary)** behavior.
    *   *Premise*: Users with consistent, high-value, and recent transaction history are lower risk.
    *   *Risk*: The risk of misclassification bias is non-trivial. A low-frequency user is not necessarily a "bad" borrower, just an inactive one. The model must carefully distinguish between "churned/inactive" and "risky."

## 2. Discussion of Completed Work and Initial Analysis (Task 1 & 2)
We have successfully set up the project structure and performed a comprehensive Exploratory Data Analysis (EDA) on the provided dataset of 95,662 transactions.

**Key Analytical Findings:**
*   **Data Quality**: The dataset is exceptionally clean with **zero missing values**, allowing us to bypass complex imputation steps.
*   **Class Imbalance**: The target variable `FraudResult` is extremely imbalanced (~0.2% positive class). This is a critical finding that dictates our modeling strategy—we must use techniques like SMOTE or cost-sensitive learning, otherwise, the model will bias heavily toward the majority class.
*   **Feature Correlations**:
    *   **Monetary Value**: `Value` (0.57) and `Amount` (0.56) are the strongest predictors of the current target (Fraud). Higher interaction values correlate with higher risk in this specific context.
    *   **Categorical Drivers**: `ProviderId` and `ChannelId` show distinct distribution patterns, suggesting that the "source" of the transaction is a strong signal for risk profiling.

## 3. Next Steps and Key Areas of Focus
**Immediate Focus (Task 3 - Feature Engineering):**
*   **RFM Construction**: We will aggregate the transaction-level data to the customer level to calculate Recency, Frequency, and Monetary scores. This will form the basis of our "Credit Risk Proxy."
*   **WoE and IV**: We will implement Weight of Evidence (WoE) binning to transform features into a linear scale suitable for scorecard development and improved interpretability.

**Future Focus (Task 4 & 5 - Modeling & MLOps):**
*   **Model Training**: We will train baseline models (Logistic Regression vs. XGBoost) to evaluate the trade-off between pure performance and interpretability.
*   **Deployment**: The final model will be served via a FastAPI endpoint, containerized with Docker, and tracked using MLFlow for reproducibility.

## 4. Report Structure and Clarity
This report follows a structured approach to document the lifecycle of the credit risk project, ensuring that technical findings are always tied back to the business objective of enabling sustainable credit products for Bati Bank.
