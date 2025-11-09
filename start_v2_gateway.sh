#!/bin/bash
# ==============================================================================
# LLaMA Gateway v2.0 Startup Script
# ==============================================================================

set -e

echo "========================================================================"
echo "  Starting EgoLlama Gateway v2.0"
echo "  ✅ FastAPI + SQLAlchemy + Redis"
echo "  ✅ PostgreSQL persistence"
echo "  ✅ GPU acceleration (NVIDIA/AMD/CPU)"
echo "========================================================================"
echo ""

# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the gateway
exec python simple_llama_gateway_crash_safe.py
