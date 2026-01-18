# =============================================================================
# Jasper ML Service Dockerfile
# =============================================================================
# Multi-stage build for smaller production images
#
# Build: docker build -t jasper-ml-service .
# Run:   docker run -p 8080:8080 jasper-ml-service
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch CPU-only (much smaller than GPU version)
# Using --index-url for CPU-only wheels
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Copy and install package
COPY pyproject.toml .
COPY ml_service/ ./ml_service/

# Install the package (without dev dependencies)
RUN pip install --no-cache-dir .


# -----------------------------------------------------------------------------
# Stage 2: Production - Minimal runtime image
# -----------------------------------------------------------------------------
FROM python:3.11-slim as production

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY ml_service/ ./ml_service/

# Copy trained model if it exists (optional - can be mounted or downloaded)
COPY models/ ./models/ 2>/dev/null || true

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Cloud Run uses PORT environment variable
ENV PORT=8080
EXPOSE 8080

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Start the FastAPI server
CMD ["python", "-m", "ml_service.api.main"]
