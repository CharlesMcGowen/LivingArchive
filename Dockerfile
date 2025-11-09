# ==============================================================================
# EgoLlama Gateway - Dockerfile
# ==============================================================================
# FastAPI Gateway with PostgreSQL and Redis support
# Supports NVIDIA CUDA, AMD ROCm, and CPU fallback
# ==============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

LABEL maintainer="EgoLlama Team"
LABEL description="LLaMA Gateway with FastAPI, PostgreSQL, Redis, and GPU support"
LABEL version="2.0.0"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    postgresql-client \
    libpq-dev \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p /app/logs && \
    chmod 777 /app/logs

# Make startup script executable
RUN chmod +x start_v2_gateway.sh || true

# Expose gateway port
EXPOSE 8082

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8082/health || exit 1

# Run the gateway
CMD ["python", "simple_llama_gateway_crash_safe.py"]








