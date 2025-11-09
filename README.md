# EgoLlama Gateway

A production-ready LLaMA Gateway with FastAPI, PostgreSQL, Redis caching, and GPU acceleration support (NVIDIA CUDA, AMD ROCm, CPU fallback).

## Features

- 🚀 **FastAPI Gateway** - Modern, async Python web framework
- 💾 **PostgreSQL** - Persistent storage for conversation history and metrics
- ⚡ **Redis** - High-performance caching and rate limiting
- 🎯 **GPU Support** - NVIDIA CUDA, AMD ROCm, and CPU fallback
- 🔄 **Graceful Degradation** - Works even if Redis/PostgreSQL are unavailable
- 📊 **Health Checks** - Built-in health monitoring endpoints
- 🐳 **Docker Ready** - Complete Docker Compose setup

## Quick Start

### ⭐ Easiest Method (Recommended)

**Difficulty:** ⭐ Easy | **Time:** 5 minutes

Just run the automated setup script:

```bash
git clone <repository-url>
cd EgoLlama-clean
./setup.sh
```

That's it! The script will:
- ✅ Check prerequisites (Docker, Docker Compose)
- ✅ Create environment configuration
- ✅ Start all services (PostgreSQL, Redis, Gateway)
- ✅ Wait for services to be healthy
- ✅ Test the gateway

### Manual Setup

**Difficulty:** ⭐ Easy | **Time:** 5-10 minutes

1. **Prerequisites**
   - Docker and Docker Compose installed
   - (Optional) NVIDIA GPU with CUDA drivers or AMD GPU with ROCm

2. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd EgoLlama-clean
   ```

3. **Configure Environment**
   ```bash
   cp env.example .env
   # Edit .env if needed (defaults work for testing)
   ```

4. **Start Services**
   ```bash
   docker-compose up -d
   ```
   This will start:
   - PostgreSQL database on port `5432`
   - Redis cache on port `6379`
   - LLaMA Gateway on port `8082`

5. **Verify Installation**
   ```bash
   # Check gateway health
   curl http://localhost:8082/health
   
   # Check database health
   curl http://localhost:8082/health/db
   
   # Check Redis health
   curl http://localhost:8082/health/redis
   ```

### Setup Difficulty Guide

For detailed difficulty assessment and setup options, see [SETUP_GUIDE.md](SETUP_GUIDE.md):
- ⭐ **Docker (Easy)** - 5-10 minutes - Recommended
- ⭐⭐ **Local Development (Moderate)** - 15-30 minutes
- ⭐⭐ **GPU Setup NVIDIA (Moderate)** - 10-20 minutes
- ⭐⭐⭐ **GPU Setup AMD (Advanced)** - 20-40 minutes

## API Endpoints

### Health Checks

- `GET /health` - Overall system health
- `GET /health/db` - Database connection health
- `GET /health/redis` - Redis connection health

### Models

- `GET /models` - List available models
- `POST /models/load` - Load a model
- `POST /models/unload` - Unload current model

### Inference

- `POST /v1/chat/completions` - OpenAI-compatible chat completions
- `POST /v1/completions` - OpenAI-compatible text completions

### Metrics

- `GET /metrics` - System and performance metrics
- `GET /stats` - Gateway statistics

## GPU Configuration

### NVIDIA GPU

1. Install NVIDIA Docker runtime:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. Update `docker-compose.yml`:
```yaml
gateway:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

3. Install PyTorch with CUDA in `requirements.txt`:
```
torch>=2.0.0+cu118
```

### AMD GPU

1. Install ROCm Docker runtime (see ROCm documentation)

2. Update `docker-compose.yml`:
```yaml
gateway:
  deploy:
    resources:
      reservations:
        devices:
          - driver: rocm
```

3. Install PyTorch with ROCm in `requirements.txt`:
```
torch>=2.0.0+rocm5.7
```

### CPU Only

No additional configuration needed. The gateway will automatically fall back to CPU mode.

## Development

### Local Development (without Docker)

1. Install Python 3.11+
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start PostgreSQL and Redis (or use Docker):
```bash
docker-compose up -d postgres redis
```

4. Set environment variables:
```bash
export EGOLLAMA_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/ego?sslmode=disable"
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
```

5. Run the gateway:
```bash
python simple_llama_gateway_crash_safe.py
```

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  FastAPI Gateway│  ← Port 8082
└────┬────────┬──┘
     │        │
     ▼        ▼
┌─────────┐ ┌─────────┐
│PostgreSQL│ │  Redis  │
└─────────┘ └─────────┘
     │
     ▼
┌─────────┐
│  GPU    │  ← NVIDIA/AMD/CPU
│ Module  │
└─────────┘
```

## Configuration

See `.env.example` for all available configuration options:

- Database connection pool settings
- Redis connection settings
- Cache TTL values
- Rate limiting configuration
- Logging levels

## Monitoring

### Health Checks

The gateway provides comprehensive health checks:

```bash
# Overall health
curl http://localhost:8082/health

# Database health
curl http://localhost:8082/health/db

# Redis health
curl http://localhost:8082/health/redis
```

### Metrics

```bash
# System metrics
curl http://localhost:8082/metrics

# Gateway statistics
curl http://localhost:8082/stats
```

## Troubleshooting

### Gateway won't start

1. Check logs:
```bash
docker-compose logs gateway
```

2. Verify database connection:
```bash
docker-compose exec postgres psql -U postgres -d ego -c "SELECT 1;"
```

3. Verify Redis connection:
```bash
docker-compose exec redis redis-cli ping
```

### GPU not detected

1. Check GPU availability:
```bash
docker-compose exec gateway python -c "import torch; print(torch.cuda.is_available())"
```

2. Verify Docker GPU runtime:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Database connection errors

1. Ensure PostgreSQL is running:
```bash
docker-compose ps postgres
```

2. Check connection string in `.env`:
```bash
cat .env | grep EGOLLAMA_DATABASE_URL
```

## License

**EgoLlama Gateway - Non-Commercial Developer License**

This software is licensed under a custom Non-Commercial Developer License that:
- ✅ **Allows free use** for individual developers, non-profit organizations, and educational purposes
- ❌ **Prohibits commercial use** without explicit written permission
- ❌ **Prohibits profit generation** from the software without permission

**Key Points:**
- Developers can use, modify, and distribute for non-commercial purposes
- Companies and for-profit entities must obtain explicit permission before commercial use
- Attribution and copyright notices must be retained

See `LICENSE` file for complete terms and conditions.

**For Commercial Licensing:** Contact the project maintainers for commercial use permissions.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Support

For issues and questions, please open a GitHub issue.

