#!/bin/bash
# ==============================================================================
# Start LivingArchive-clean Gateway in Standalone Mode
# ==============================================================================
# Bypasses Docker and runs directly with Python
# ==============================================================================

set -e

echo "============================================================================"
echo "  Starting LivingArchive-clean Gateway (Standalone Mode)"
echo "============================================================================"
echo ""

# Check Python
if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if dependencies are installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "⚠️  FastAPI not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Set default environment variables if not set
export GATEWAY_PORT=${GATEWAY_PORT:-7775}
export GATEWAY_HOST=${GATEWAY_HOST:-0.0.0.0}
export EGOLLAMA_PORT=${EGOLLAMA_PORT:-7775}
export EGOLLAMA_HOST=${EGOLLAMA_HOST:-0.0.0.0}

echo "✅ Starting gateway on http://${GATEWAY_HOST}:${GATEWAY_PORT}"
echo ""
echo "💡 Note: PostgreSQL and Redis are optional - gateway will work without them"
echo "   For full features, start PostgreSQL and Redis separately"
echo ""
echo "📝 To use a different port, set GATEWAY_PORT environment variable:"
echo "   export GATEWAY_PORT=7775 && ./start_standalone.sh"
echo ""

# Run the gateway
exec python3 simple_llama_gateway_crash_safe.py

