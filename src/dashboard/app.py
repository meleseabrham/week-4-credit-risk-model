import streamlit as st
import requests
import datetime

# Define API URL
API_URL = "http://api:8000/predict"  # Using docker service name "api"

st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Risk Scoring Dashboard")
st.markdown("### Evaluate Customer Creditworthiness")

# Input Form
with st.form("risk_form"):
    st.subheader("Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input(
            "Transaction Amount", min_value=0.0, value=1000.0, step=100.0
        )
        val = st.number_input(
            "Transaction Value", min_value=0.0, value=1000.0, step=100.0
        )
        provider_id = st.text_input("Provider ID", value="Provider_1")
        channel_id = st.selectbox(
            "Channel ID", ["Web", "Android", "iOS", "PayLater"]
        )

    with col2:
        product_id = st.text_input("Product ID", value="Product_1")
        product_category = st.selectbox(
            "Product Category",
            ["Financial Services", "Airtime", "DataBundles", "UtilityBill"]
        )
        pricing_strategy = st.text_input("Pricing Strategy", value="Tier_1")
        customer_id = st.text_input("Customer ID", value="Cust_001")

    transaction_date = st.date_input("Transaction Date", datetime.date.today())
    transaction_time = st.time_input("Transaction Time", datetime.time(10, 0))

    # Construct TransactionStartTime using f-string for ISO 8601 format
    transaction_start_time = f"{transaction_date}T{transaction_time}Z"

    submit_button = st.form_submit_button("Predict Risk")

if submit_button:
    # Prepare payload
    payload = {
        "Amount": amount,
        "Value": val,
        "TransactionStartTime": transaction_start_time,
        "CustomerId": customer_id,
        "ProviderId": provider_id,
        "ProductId": product_id,
        "ProductCategory": product_category,
        "ChannelId": channel_id,
        "PricingStrategy": pricing_strategy
    }

    with st.spinner("Analyzing Risk..."):
        try:
            # When running in Docker, we use the service name.
            # If running locally outside docker, user might need to change
            # this or use localhost.
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                result = response.json()
                prob = result["probability"]
                is_high_risk = result["is_high_risk"]

                st.divider()
                st.subheader("Results")

                col_res1, col_res2 = st.columns(2)

                with col_res1:
                    st.metric(label="Risk Probability", value=f"{prob:.2%}")

                with col_res2:
                    if is_high_risk:
                        st.error("🚨 High Risk Customer")
                    else:
                        st.success("✅ Low Risk Customer")

                st.progress(prob, text="Risk Score")

            else:
                st.error(f"Error: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Failed to connect to the scoring API. "
                "Ensure the API service is running."
            )
