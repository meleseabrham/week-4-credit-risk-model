"""
Comprehensive API tests using FastAPI TestClient.

Tests cover request validation, successful predictions,
error handling, and edge cases.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main_enhanced import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "running"


def test_health_endpoint():
    """Test the detailed health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_valid_prediction_request():
    """Test a valid prediction request."""
    payload = {
        "Amount": 1000.0,
        "Value": 1000.0,
        "TransactionStartTime": "2023-01-01T12:00:00Z",
        "CustomerId": "C001",
        "ProviderId": "P001",
        "ProductId": "PR001",
        "ProductCategory": "Financial Services",
        "ChannelId": "Web",
        "PricingStrategy": "Tier_1"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "probability" in data
    assert "is_high_risk" in data
    assert 0 <= data["probability"] <= 1
    assert data["is_high_risk"] in [0, 1]


def test_high_amount_transaction():
    """Test prediction for high amount transaction."""
    payload = {
        "Amount": 10000.0,
        "Value": 10000.0,
        "TransactionStartTime": "2023-01-01T12:00:00Z",
        "CustomerId": "C002",
        "ProviderId": "P001",
        "ProductId": "PR001",
        "ProductCategory": "Financial Services",
        "ChannelId": "Web",
        "PricingStrategy": "Tier_1"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    # High amounts might indicate higher risk in dummy logic
    assert "probability" in data


def test_low_amount_transaction():
    """Test prediction for low amount transaction."""
    payload = {
        "Amount": 100.0,
        "Value": 100.0,
        "TransactionStartTime": "2023-01-01T12:00:00Z",
        "CustomerId": "C003",
        "ProviderId": "P001",
        "ProductId": "PR001",
        "ProductCategory": "Airtime",
        "ChannelId": "Mobile",
        "PricingStrategy": "Standard"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["probability"] >= 0


def test_missing_required_field():
    """Test request with missing required field."""
    payload = {
        "Amount": 1000.0,
        # Missing "Value"
        "TransactionStartTime": "2023-01-01T12:00:00Z",
        "CustomerId": "C004",
        "ProviderId": "P001",
        "ProductId": "PR001",
        "ProductCategory": "Financial Services",
        "ChannelId": "Web",
        "PricingStrategy": "Tier_1"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_invalid_amount_type():
    """Test request with invalid data type."""
    payload = {
        "Amount": "invalid",  # Should be float
        "Value": 1000.0,
        "TransactionStartTime": "2023-01-01T12:00:00Z",
        "CustomerId": "C005",
        "ProviderId": "P001",
        "ProductId": "PR001",
        "ProductCategory": "Financial Services",
        "ChannelId": "Web",
        "PricingStrategy": "Tier_1"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_invalid_datetime_format():
    """Test request with invalid datetime format."""
    payload = {
        "Amount": 1000.0,
        "Value": 1000.0,
        "TransactionStartTime": "invalid-datetime",
        "CustomerId": "C006",
        "ProviderId": "P001",
        "ProductId": "PR001",
        "ProductCategory": "Financial Services",
        "Channeld": "Web",
        "PricingStrategy": "Tier_1"
    }

    response = client.post("/predict", json=payload)
    # May fail validation or return 500 depending on handling
    assert response.status_code in [422, 500]


def test_negative_amount():
    """Test handling of edge case: negative amount."""
    payload = {
        "Amount": -100.0,
        "Value": 1000.0,
        "TransactionStartTime": "2023-01-01T12:00:00Z",
        "CustomerI": "C007",
        "ProviderId": "P001",
        "ProductId": "PR001",
        "ProductCategory": "Financial Services",
        "ChannelId": "Web",
        "PricingStrategy": "Tier_1"
    }

    response = client.post("/predict", json=payload)
    # Should either validate or handle gracefully
    assert response.status_code in [200, 422]


def test_multiple_predictions_consistency():
    """Test that same input gives consistent predictions."""
    payload = {
        "Amount": 1500.0,
        "Value": 1500.0,
        "TransactionStartTime": "2023-06-15T10:30:00Z",
        "CustomerId": "C008",
        "ProviderId": "P002",
        "ProductId": "PR002",
        "ProductCategory": "DataBundles",
        "ChannelId": "Android",
        "PricingStrategy": "Premium"
    }

    response1 = client.post("/predict", json=payload)
    response2 = client.post("/predict", json=payload)

    assert response1.status_code == 200
    assert response2.status_code == 200

    data1 = response1.json()
    data2 = response2.json()

    # Same input should give same output
    assert data1["probability"] == data2["probability"]
    assert data1["is_high_risk"] == data2["is_high_risk"]


def test_various_product_categories():
    """Test predictions across different product categories."""
    categories = [
        "Financial Services",
        "Airtime",
        "DataBundles",
        "UtilityBill"
    ]

    for category in categories:
        payload = {
            "Amount": 1000.0,
            "Value": 1000.0,
            "TransactionStartTime": "2023-01-01T12:00:00Z",
            "CustomerId": f"C_{category}",
            "ProviderId": "P001",
            "ProductId": "PR001",
            "ProductCategory": category,
            "ChannelId": "Web",
            "PricingStrategy": "Standard"
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "probability" in data
        assert "is_high_risk" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
