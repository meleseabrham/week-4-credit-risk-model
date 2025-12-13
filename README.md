# Credit Risk Scoring Model

An end-to-end Machine Learning project to assess creditworthiness using transaction data. This system utilizes a proxy target variable derived from RFM (Recency, Frequency, Monetary) analysis to predict the probability of default/high risk.

## 🚀 Project Overview
**Bati Bank** has partnered with an eCommerce platform to enable a Buy-Now-Pay-Later service. This project builds a Credit Scoring Model to estimate the likelihood of a customer defaulting.

**Key Components:**
*   **Feature Engineering**: Custom transformers for Aggregate, Time-Series, WoE, and RFM features.
*   **Target Engineering**: K-Means clustering on RFM metrics to label users as High/Low Risk.
*   **Model Training**: scikit-learn pipeline (Logistic Regression/Random Forest) with **MLflow** for experiment tracking.
*   **Deployment**: **FastAPI** service for real-time inference.
*   **User Interface**: **Streamlit** dashboard for interactive risk assessment.
*   **Infrastructure**: Fully containerized with **Docker** & **Docker Compose**.
*   **CI/CD**: Automatic linting and testing via GitHub Actions.

---

## 📂 Project Structure
```
credit-risk-model/
├── .github/workflows/ # CI/CD configurations
├── data/              # Raw and processed datasets (git-ignored)
├── notebooks/         # Jupyter notebooks for EDA and prototyping
├── src/
│   ├── api/           # FastAPI application
│   ├── dashboard/     # Streamlit UI
│   ├── data_processing.py # Feature engineering pipeline
│   ├── train.py       # Training script with MLflow
│   └── ...
├── tests/             # Unit tests (pytest)
├── Dockerfile         # API image definition
├── Dockerfile.streamlit # Dashboard image definition
├── docker-compose.yml # Service orchestration
└── requirements.txt   # Python dependencies
```

---

## 🛠️ Setup & Installation

### Option A: Running with Docker (Recommended)
You can bring up the entire stack (API + Dashboard + MLflow Tracking) with a single command.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/meleseabrham/week-4-credit-risk-model.git
    cd week-4-credit-risk-model
    ```
2.  **Build and Run**:
    ```bash
    docker-compose up --build
    ```
3.  **Access Services**:
    *   **Dashboard (UI)**: [http://localhost:8501](http://localhost:8501)
    *   **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
    *   **MLflow UI** (if configured locally): Check logs for tracking URI.

### Option B: Local Development
1.  **Create a virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃‍♂️ How to Run

### 1. Train the Model
The training script loads data, preprocesses it, trains models, and logs results to MLflow.
```bash
python src/train.py
```
*Artifacts (models/metrics) will be saved to `./mlruns` and `./models`.*

### 2. Run the API API
```bash
uvicorn src.api.main:app --reload
```
Test prediction:
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"Amount": 1000, "Value": 1000, "TransactionStartTime": "2023-01-01T12:00:00Z", "CustomerId": "C1", "ProviderId": "P1", "ProductId": "Pr1", "ProductCategory": "Cat1", "ChannelId": "Web", "PricingStrategy": "PlanA"}'
```

### 3. Run the Dashboard
```bash
streamlit run src/dashboard/app.py
```

---

## 🧠 Business Understanding & Methodology

### Basel II Accord Significance
The Basel II Capital Accord mandates rigorous risk measurement. Our model focuses on **interpretability** and consistency to ensure that risk scores are auditable and justifiable for capital allocation decisions.

### Proxy Target Variable (RFM)
Since no direct "default" label exists, we engineered a proxy:
1.  **RFM Calculation**: Computed Recency, Frequency, and Monetary value for each user.
2.  **Clustering**: Used K-Means to identify a "High Risk" cluster (typically low frequency/low monetary value groups).
3.  **Labeling**: Users in this cluster are flagged as `is_high_risk = 1`.

### Model Trade-offs
*   **Logistic Regression**: Selected for its **high interpretability** and regulatory friendliness (Scorecards).
*   **Random Forest / GBM**: Used as a performance benchmark to capture non-linear complex patterns.

---

## 📊 Exploratory Data Analysis Insights
*   **Data Completeness**: Dataset is 100% complete.
*   **Class Imbalance**: Fraud/Risk events are rare (~0.2% for fraud), necessitating robust evaluation metrics like **ROC-AUC** and **F1-Score** rather than simple Accuracy.
*   **Key Drivers**: Transaction `Value` and `Amount` are the strongest predictors of risk/fraud status.

---

## ✅ CI/CD Pipeline
Every push to `main` triggers a GitHub Actions workflow that:
1.  **Lints** the code with `flake8`.
2.  **Tests** data processing logic with `pytest`.
Builds fail if code quality standards are not met.


