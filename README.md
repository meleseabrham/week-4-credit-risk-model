# 💳 Credit Risk Scoring Model

[![CI](https://github.com/meleseabrham/week-4-credit-risk-model/workflows/CI/badge.svg)](https://github.com/meleseabrham/week-4-credit-risk-model/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-black)](https://flake8.pycqa.org/)

An end-to-end Machine Learning system to assess creditworthiness using transaction data. This production-ready solution implements a proxy target variable derived from RFM (Recency, Frequency, Monetary) analysis to predict the probability of customer default.

## 🚀 Project Overview

**Bati Bank** has partnered with an eCommerce platform to enable a Buy-Now-Pay-Later service. This project builds a complete Credit Scoring pipeline to estimate the likelihood of a customer defaulting on credit obligations.

### Key Components

*   **Advanced Feature Engineering**: Custom sklearn transformers for Aggregate, Time-Series, and RFM features with full pipeline integration
*   **Proxy Target Engineering**: K-Means clustering on RFM metrics with documented rationale and diagnostics
*   **Model Training & Tracking**: scikit-learn models with comprehensive MLflow experiment tracking and Model Registry integration
*   **Production Deployment**: FastAPI service loading actual trained models with proper preprocessing
*   **Interactive UI**: Streamlit dashboard for non-technical users
*   **Infrastructure**: Fully containerized with Docker & Docker Compose
* **CI/CD**: Automated linting, testing, and quality checks via GitHub Actions

---

## 📂 Project Structure

```
credit-risk-model/
├── .github/workflows/
│   └── ci.yml                    # CI/CD pipeline configuration
├── data/
│   ├── raw/                      # Raw data (git-ignored)
│   └── processed/                # Processed datasets (git-ignored)
├── notebooks/
│   └── eda.ipynb                 # Exploratory Data Analysis
├── src/
│   ├── api/
│   │   ├── main.py               # Basic FastAPI application
│   │   ├── main_enhanced.py      # Production API with MLflow integration
│   │   └── pydantic_models.py    # Request/response validation
│   ├── dashboard/
│   │   └── app.py                # Streamlit UI
│   ├── data_processing.py        # Original preprocessing
│   ├── preprocessing_pipeline.py # Enhanced sklearn Pipeline
│   ├── train.py                  # Basic training script
│   └── train_enhanced.py         # Production training with Model Registry
├── tests/
│   ├── test_data_processing.py   # Unit tests for preprocessing
│   └── test_api.py               # API integration tests
├── models/                       # Saved models (git-ignored)
├── mlruns/                       # MLflow tracking data (git-ignored)
├── Dockerfile                    # API container
├── Dockerfile.streamlit          # Dashboard container
├── docker-compose.yml            # Service orchestration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

##🛠️ Setup & Installation

### Prerequisites

- **Docker** (recommended) OR
- **Python 3.9+** for local development
- **Git** for version control

### Option A: Running with Docker (Recommended)

Deploy the entire stack (API + Dashboard + MLflow) with a single command:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/meleseabrham/week-4-credit-risk-model.git
    cd week-4-credit-risk-model
    ```

2.  **Build and run**:
    ```bash
    docker-compose up --build -d
    ```

3.  **Access services**:
    *   📊 **Dashboard (UI)**: [http://localhost:8501](http://localhost:8501)
    *   🔧 **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
    *   📈 **MLflow UI**: Configure tracking URI in docker-compose

4.  **Stop services**:
    ```bash
    docker-compose down
    ```

### Option B: Local Development

1.  **Create virtual environment** (Windows):
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate
    ```

    **For Linux/Mac**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃‍♂️ How to Run

### 1. Train the Model

Run the enhanced training script with comprehensive tracking:

```bash
python src/train_enhanced.py
```

**What it does:**
- Loads and preprocesses data
- Trains multiple models (Logistic Regression, Random Forest)
- Performs 5-fold cross-validation
- Logs metrics, parameters, and artifacts to MLflow
- Registers the best model in MLflow Model Registry
- Saves model locally to `models/best_model.pkl`

**MLflow artifacts logged:**
- Feature importance / coefficients
- Cross-validation metrics per fold
- Model parameters
- Evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)

### 2. Run the API

**Production API (with MLflow model loading)**:
```bash
uvicorn src.api.main_enhanced:app --reload
```

**Basic API (original)**:
```bash
uvicorn src.api.main:app --reload
```

**Test prediction**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Amount": 1000,
    "Value": 1000,
    "TransactionStartTime": "2023-01-01T12:00:00Z",
    "CustomerId": "C001",
    "ProviderId": "P001",
    "ProductId": "PR001",
    "ProductCategory": "Financial Services",
    "ChannelId": "Web",
    "PricingStrategy": "Standard"
  }'
```

### 3. Run the Dashboard

```bash
streamlit run src/dashboard/app.py
```

Navigate to [http://localhost:8501](http://localhost:8501) to interact with the UI.

### 4. Run Tests

**All tests**:
```bash
pytest tests/ -v
```

**Specific test suites**:
```bash
pytest tests/test_data_processing.py -v  # Preprocessing tests
pytest tests/test_api.py -v              # API integration tests
```

**With coverage**:
```bash
pytest --cov=src tests/
```

### 5. Code Quality Checks

```bash
flake8 src/ tests/ --max-line-length=88
```

---

## 🧠 Business Understanding & Methodology

### Basel II Accord & Interpretability

The Basel II Capital Accord mandates rigorous, transparent risk measurement for minimum capital requirements. Our model prioritizes:
- **Interpretability**: Clear feature importance and decision logic
- **Auditability**: Full tracking of experiments and model versions
- **Regulatory Compliance**: Explainable predictions for stakeholder review

### Proxy Target Variable (RFM Clustering)

Since no direct "default" label exists, we engineered a proxy using behavioral data:

1.  **RFM Calculation**:
    - **Recency**: Days since last transaction
    - **Frequency**: Total transaction count per customer
    - **Monetary**: Total transaction amount per customer

2.  **K-Means Clustering (n_clusters=3)**:
    - **Rationale**: Three segments align with standard credit grades (A/B/C or Low/Med/High risk)
    - **High-Risk Definition**: Cluster with lowest `TotalTransactionAmount` indicates minimal engagement → higher credit risk
    - **Diagnostics Logged**: Cluster centers, inertia, and segment statistics

3.  **Labeling**: Customers in the high-risk cluster receive `is_high_risk = 1`

### Model Selection Trade-offs

| Model | Interpretability | Performance | Regulatory Fit |
|-------|-----------------|-------------|----------------|
| **Logistic Regression** | ⭐⭐⭐⭐⭐ High (clear coefficients) | ⭐⭐⭐ Moderate (linear assumptions) | ⭐⭐⭐⭐⭐ Excellent (industry standard) |
| **Random Forest** | ⭐⭐⭐ Moderate (feature importance) | ⭐⭐⭐⭐⭐ High (captures nonlinearity) | ⭐⭐⭐ Good (with SHAP/LIME) |

**Strategy**: We train both and select based on F1-score, prioritizing interpretability when performance is comparable.

---

## 📊 Exploratory Data Analysis Insights

*   **Data Completeness**: 100% complete dataset (no missing values)
*   **Class Imbalance**: Proxy target shows ~20-30% high-risk customers
*   **Key Predictors**:
    - Transaction `Amount` and `Value` (strong correlation with risk)
    - `TransactionCount` and `Recency` (behavioral indicators)
    - `ProductCategory` and `ChannelId` (categorical patterns)

*   **Evaluation Metrics**: We prioritize ROC-AUC and F1-Score over Accuracy due to potential class imbalance

---

## ✅ CI/CD Pipeline

Every push to `main` triggers automated checks:

1.  **Code Linting** (`flake8`): Enforces PEP 8 style guidelines
2.  **Unit Tests** (`pytest`): Validates preprocessing and API logic
3.  **Build Verification**: Ensures Docker images build successfully

Builds **fail** if:
- Linting errors exist
- Any test fails
- Code coverage drops below threshold (if configured)

---

## 🔒 Git & GitHub Best Practices

### Branching Strategy
- `main`: Production-ready code
- `task-*`: Feature branches for specific tasks
- `hotfix-*`: Emergency fixes

### Commit Message Convention
```
<type>(<scope>): <subject>

Examples:
feat(api): add MLflow model loading
fix(preprocessing): handle missing timestamps
docs(readme): update installation instructions
test(api): add edge case validation tests
```

### Pull Request Template
- **Problem**: What issue does this solve?
- **Solution**: How did you solve it?
- **Testing**: What tests were added/run?
- **Screenshots**: (if UI changes)

### Branch Protection Rules (Recommended)
- Require PR reviews before merging
- Require status checks to pass (CI green)
- Enforce linear history
- Restrict force pushes

---

## 🚦 Production Deployment Checklist

- [x] Model trained and registered in MLflow
- [x] API loads model from MLflow/local storage
- [x] Preprocessing pipeline matches training exactly
- [x] Comprehensive API tests (validation, errors, edge cases)
- [x] Docker containers build and run successfully
- [x] CI/CD pipeline passing
- [x] Code style compliant (flake8)
- [x] Documentation complete
- [ ] Load testing performed (recommended)
- [ ] Monitoring/logging configured (recommended)
- [ ] Security review completed (recommended)

---

## 📈 Performance Metrics

| Metric | Logistic Regression | Random Forest |
|--------|---------------------|---------------|
| Accuracy | ~0.XX | ~0.XX |
| Precision | ~0.XX | ~0.XX |
| Recall | ~0.XX | ~0.XX |
| F1-Score | ~0.XX | ~0.XX |
| ROC-AUC | ~0.XX | ~0.XX |

*(Run training to populate actual metrics)*

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👥 Authors

- **Melese Abrham** - [GitHub](https://github.com/meleseabrham)

---

## 🙏 Acknowledgments

- 10 Academy for project guidance
- Bati Bank for the business case
- MLflow community for experiment tracking tools
- FastAPI and Streamlit for modern Python frameworks

---

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Contact: [your-email@example.com]

---

**Built with ❤️ using Python, scikit-learn, MLflow, FastAPI, and Docker**
