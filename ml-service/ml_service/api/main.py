"""
FastAPI application for Jasper image classification.

Entry point: jasper-serve

Endpoints:
    GET  /           - API info
    GET  /health     - Health check (required for Cloud Run)
    POST /predict    - Image classification
"""

import io
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from ml_service.models.classifier import create_model, load_model, get_device


# ============================================================================
# Pydantic Models (Request/Response schemas)
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]


class APIInfoResponse(BaseModel):
    name: str
    version: str
    description: str
    endpoints: list[str]


# ============================================================================
# App Configuration
# ============================================================================

app = FastAPI(
    title="Jasper Classifier API",
    description="ML service to identify Jasper (a white dog) from images",
    version="0.1.0",
)

# CORS middleware - allows frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Global State
# ============================================================================

# Model and device are loaded once at startup
model: Optional[torch.nn.Module] = None
device: Optional[torch.device] = None
CLASS_NAMES = ["not_jasper", "jasper"]

# Image preprocessing - must match training transforms
inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def load_model_on_startup():
    """Load the model when the server starts."""
    global model, device

    device = get_device()
    model_path = Path(__file__).parent.parent.parent / "models" / "jasper_classifier.pth"

    if model_path.exists():
        print(f"Loading trained model from {model_path}")
        model, metadata = load_model(model_path)
        model = model.to(device)
        model.eval()
        print(f"Model loaded. Metadata: {metadata}")
    else:
        print(f"No trained model found at {model_path}")
        print("Using placeholder model (random predictions)")
        # Create a fresh model for placeholder predictions
        model = create_model(num_classes=2, freeze_backbone=True)
        model = model.to(device)
        model.eval()


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", response_model=APIInfoResponse)
async def root():
    """API information endpoint."""
    return APIInfoResponse(
        name="Jasper Classifier API",
        version="0.1.0",
        description="Upload an image to check if it contains Jasper",
        endpoints=["/", "/health", "/predict"],
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Cloud Run uses this to verify the service is ready to receive traffic.
    Returns 200 if healthy, which tells Cloud Run the container is ready.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(device) if device else "not initialized",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Classify an uploaded image.

    Args:
        file: Image file (JPEG, PNG, etc.)

    Returns:
        Prediction with confidence scores
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Got: {file.content_type}"
        )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service not ready."
        )

    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Apply transforms and add batch dimension
        input_tensor = inference_transform(image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        # Get prediction
        predicted_idx = probabilities.argmax().item()
        confidence = probabilities[predicted_idx].item()

        return PredictionResponse(
            prediction=CLASS_NAMES[predicted_idx],
            confidence=round(confidence, 4),
            probabilities={
                CLASS_NAMES[i]: round(probabilities[i].item(), 4)
                for i in range(len(CLASS_NAMES))
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


# ============================================================================
# Entry Point
# ============================================================================

def serve():
    """Start the FastAPI server using uvicorn."""
    import uvicorn

    # Get port from environment variable (Cloud Run sets PORT)
    import os
    port = int(os.environ.get("PORT", 8080))

    print(f"Starting Jasper Classifier API on port {port}")
    uvicorn.run(
        "ml_service.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
    )


if __name__ == "__main__":
    serve()
