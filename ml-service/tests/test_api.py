"""
Unit tests for the Jasper Classifier API.

These tests verify the API endpoints work correctly.
They use FastAPI's TestClient which doesn't require a running server.
"""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from ml_service.api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_image():
    """Create a simple test image in memory."""
    # Create a small RGB image (100x100 pixels, white)
    img = Image.new("RGB", (100, 100), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


class TestRootEndpoint:
    """Tests for GET / endpoint."""

    def test_root_returns_api_info(self, client):
        """Root endpoint should return API information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Jasper Classifier API"
        assert data["version"] == "0.1.0"
        assert "/health" in data["endpoints"]
        assert "/predict" in data["endpoints"]


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_returns_healthy(self, client):
        """Health endpoint should return healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "device" in data

    def test_health_model_loaded(self, client):
        """Health endpoint should indicate model is loaded."""
        response = client.get("/health")

        data = response.json()
        # Model should be loaded (either trained or placeholder)
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    """Tests for POST /predict endpoint."""

    def test_predict_with_valid_image(self, client, sample_image):
        """Predict endpoint should accept valid image and return prediction."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data

        # Check prediction is valid class
        assert data["prediction"] in ["jasper", "not_jasper"]

        # Check confidence is between 0 and 1
        assert 0 <= data["confidence"] <= 1

        # Check probabilities contain both classes
        assert "jasper" in data["probabilities"]
        assert "not_jasper" in data["probabilities"]

    def test_predict_with_png_image(self, client):
        """Predict endpoint should accept PNG images."""
        # Create PNG image
        img = Image.new("RGB", (100, 100), color="blue")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        response = client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
        )

        assert response.status_code == 200

    def test_predict_rejects_non_image(self, client):
        """Predict endpoint should reject non-image files."""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )

        assert response.status_code == 400
        assert "must be an image" in response.json()["detail"]

    def test_predict_requires_file(self, client):
        """Predict endpoint should require a file."""
        response = client.post("/predict")

        assert response.status_code == 422  # Validation error
