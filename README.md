# Credit Risk Model

Project for credit risk modeling.

## Structure
- `data/`: Raw and processed data
- `notebooks/`: Jupyter notebooks for EDA
- `src/`: Source code for data processing, training, and inference
- `tests/`: Unit tests

## Credit Scoring Business Understanding

### Basel II Accord and Model Interpretability
The Basel II Capital Accord emphasizes rigorous risk measurement to determine the minimum capital requirements for financial institutions. It enables banks to use their own internal estimates of risk components—Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD)—provided these systems are validated, transparent, and auditable. This regulatory framework necessitates models that are not only accurate but also **interpretable**. We must be able to explain *why* a customer was assigned a specific risk score to regulators and internal auditors. A "black box" model, no matter how accurate, struggles to meet these compliance standards because it lacks the traceability required to justify capital allocation decisions.

### The Need for Proxy Variables and Associated Risks
In this project, we lack a direct "default" label (historical data on who failed to pay back a loan). Therefore, we must engineer a **proxy variable** using behavioral data, specifically Recency, Frequency, and Monetary (RFM) patterns, to categorize users as "high risk" (bad) or "low risk" (good).
**Why:** We assume that stable, frequent, and high-value transactional behavior correlates with financial stability and willingness to repay.
**Risks:** The primary risk is **misclassification bias**. A customer with low transaction volume on our platform might simply prefer cash or other platforms, not necessarily be a credit risk. Conversely, a high-volume user might be over-leveraged. If our proxy is flawed, the model will learn to predict the *proxy*, not actual creditworthiness, leading to bad loans (financial loss) or rejected good customers (lost revenue).

### Model Selection Trade-offs: Logistic Regression vs. Gradient Boosting
In a regulated financial context, the choice between simple and complex models is a critical trade-off:

| Feature | Logistic Regression (with WoE) | Gradient Boosting (e.g., XGBoost/CatBoost) |
| :--- | :--- | :--- |
| **Interpretability** | **High**. Coefficients directly translate to Odds Ratios. Easy to convert into a traditional Scorecard (points system). | **Low**. "Black box" nature requires secondary explainability tools (SHAP, LIME) to understand feature impact. |
| **Performance** | Moderate. Assumes linear relationships (unless transformed). | **High**. Captures complex non-linear patterns and interactions automatically. |
| **Regulatory Fit** | **Excellent**. The industry standard for decades. Easy to audit and justify. | **Challenging**. Requires rigorous validation and robust explainability documentation to satisfy regulators. |

**Decision:** We will likely aim for a balance—using Gradient Boosting to establish a performance benchmark, but potentially prioritizing a simpler, scorecard-compatible model for the final deployment if regulatory constraints are strict.

