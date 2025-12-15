"""
Comprehensive API tests using FastAPI TestClient.

Tests cover request validation, successful predictions,
and error handling.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
