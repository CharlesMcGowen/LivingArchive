# LivingArchive

> Production-ready API gateway for Large Language Models (LLMs) with OpenAI-compatible endpoints

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)

LivingArchive (formerly EgoLlama Gateway) is a comprehensive API gateway that provides a unified interface for running Large Language Models. It offers OpenAI-compatible endpoints, supports multiple model backends (Ollama, HuggingFace), and includes enterprise-grade features like Redis caching, PostgreSQL persistence, and security controls.

## ✨ Features

- 🤖 **OpenAI-Compatible API** - Drop-in replacement for OpenAI API endpoints
- 🦙 **Multiple Model Support** - Ollama, HuggingFace, and custom model backends
- 🔒 **Enterprise Security** - API key authentication, CORS, rate limiting
- ⚡ **High Performance** - Redis caching, request batching, optimized inference
- 🐳 **Docker Ready** - Easy deployment with Docker Compose
- 💾 **PostgreSQL Persistence** - Store inference history and metadata
- 📊 **Monitoring & Metrics** - Health checks, performance stats, usage tracking
- 🔄 **Graceful Degradation** - Works with or without optional dependencies
- 🌐 **WebSocket Support** - Streaming responses for real-time interactions
- 🛠️ **Production Ready** - Comprehensive error handling, logging, and recovery

## 🚀 Quick Start

### Option 1: One-Command Quick Start (Recommended)

The easiest way to get started:

```bash
./quick_start.sh
```

This script automatically:
- ✅ Detects your environment (Docker or standalone)
- ✅ Sets up all dependencies
- ✅ Configures environment variables
- ✅ Starts required services
- ✅ Tests deployment

### Option 2: Docker Deployment

```bash
# Clone the repository
git clone https://github.com/CharlesMcGowen/LivingArchive.git
cd LivingArchive

# Configure environment
cp env.example .env
# Edit .env with your settings

# Start services
docker-compose up -d

# Verify installation
curl http://localhost:8082/health
```

### Option 3: Standalone Deployment

```bash
# Run deployment script
./deploy_standalone.sh

# Start the gateway
./start_standalone.sh
```

## 📋 Prerequisites

### Docker Deployment
- Docker 20.10+
- Docker Compose 2.0+

### Standalone Deployment
- Python 3.11+
- pip
- (Optional) PostgreSQL 14+
- (Optional) Redis 6+

## 🔧 Installation

### Docker Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/CharlesMcGowen/LivingArchive.git
   cd LivingArchive
   ```

2. **Configure environment:**
   ```bash
   cp env.example .env
   # Edit .env file with your configuration
   ```

3. **Start services:**
   ```bash
   docker-compose up -d
   ```

4. **Verify installation:**
   ```bash
   curl http://localhost:8082/health
   ```

### Manual Installation

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp env.example .env
   # Edit .env file
   ```

4. **Initialize database (optional):**
   ```bash
   alembic upgrade head
   ```

5. **Start the gateway:**
   ```bash
   python simple_llama_gateway_crash_safe.py
   ```

## 🎯 Usage Examples

### Basic Text Generation

```bash
curl -X POST http://localhost:8082/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "prompt": "Explain quantum computing in simple terms",
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

### Chat Completions (OpenAI-Compatible)

```bash
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "model": "mistral:7b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is Python?"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### Python Client Example

```python
import requests

url = "http://localhost:8082/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "your-api-key"
}
data = {
    "model": "mistral:7b",
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
}

response = requests.post(url, json=data, headers=headers)
result = response.json()
print(result["choices"][0]["message"]["content"])
```

### JavaScript/Node.js Example

```javascript
const fetch = require('node-fetch');

async function chatCompletion(messages) {
  const response = await fetch('http://localhost:8082/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': 'your-api-key'
    },
    body: JSON.stringify({
      model: 'mistral:7b',
      messages: messages,
      max_tokens: 256
    })
  });
  
  const result = await response.json();
  return result.choices[0].message.content;
}
```

## 📚 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate text from a prompt |
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions |
| `/v1/completions` | POST | OpenAI-compatible text completions |
| `/health` | GET | Health check (public) |
| `/api/status` | GET | API status and features |

### Model Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models` | GET | List available models |
| `/models/load` | POST | Load a model into memory |
| `/models/unload` | POST | Unload current model |
| `/api/ollama/models` | GET | List Ollama models |
| `/api/ollama/pull` | POST | Pull an Ollama model |

### Monitoring & Stats

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/performance/stats` | GET | Performance statistics |
| `/api/cache/stats` | GET | Cache statistics |
| `/api/db/health` | GET | Database health check |
| `/api/redis/health` | GET | Redis health check |

For detailed API documentation, see [WIKI.md](WIKI.md).

## ⚙️ Configuration

### Environment Variables

Key configuration options (see `env.example` for complete list):

```bash
# Security (Required for production)
EGOLLAMA_API_KEY=your-strong-secret-key-here
EGOLLAMA_REQUIRE_API_KEY=true
EGOLLAMA_CORS_ORIGINS=https://yourdomain.com

# Server
EGOLLAMA_HOST=0.0.0.0
EGOLLAMA_PORT=8082
ENVIRONMENT=production

# Database (Optional)
EGOLLAMA_DATABASE_URL=postgresql://user:pass@localhost:5432/ego

# Redis (Optional)
REDIS_HOST=localhost
REDIS_PORT=6379

# Model Backend
OLLAMA_BASE_URL=http://localhost:11434
HF_TOKEN=your-huggingface-token  # Optional, for private models
```

### Production Configuration

For production deployments:

1. **Set environment mode:**
   ```bash
   ENVIRONMENT=production
   ```

2. **Configure API key:**
   ```bash
   EGOLLAMA_API_KEY=$(openssl rand -hex 32)
   EGOLLAMA_REQUIRE_API_KEY=true
   ```

3. **Restrict CORS:**
   ```bash
   EGOLLAMA_CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
   ```

4. **Security settings:**
   ```bash
   EGOLLAMA_TRUST_REMOTE_CODE=false  # Keep false for security
   ```

For detailed configuration guide, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

## 🦙 Model Setup

### Installing Ollama Models

1. **Automatic setup:**
   ```bash
   ./setup_ollama.sh
   ```

2. **Manual setup:**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama service
   ollama serve &
   
   # Pull models
   ollama pull mistral:7b
   ollama pull llama3.1:8b
   ollama pull codellama:7b
   ```

### Using HuggingFace Models

Models are automatically downloaded from HuggingFace Hub when requested:

```bash
curl -X POST http://localhost:8082/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Hello!",
    "max_tokens": 100
  }'
```

## 🏗️ Architecture

```
┌─────────────────┐
│   Client Apps   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  LivingArchive Gateway  │
│   (FastAPI Server)      │
├─────────────────────────┤
│  • Authentication       │
│  • Rate Limiting        │
│  • Request Routing      │
│  • Response Caching     │
└────────┬────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌──────────┐
│ Ollama  │ │HuggingFace│
│ Models  │ │  Models   │
└─────────┘ └──────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌──────────┐
│PostgreSQL│ │  Redis   │
│ Database │ │  Cache   │
└─────────┘ └──────────┘
```

## 📖 Documentation

- **[WIKI.md](WIKI.md)** - Complete API documentation and usage guide
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Detailed deployment instructions
- **[SECURITY_AUDIT_REPORT.md](SECURITY_AUDIT_REPORT.md)** - Security assessment
- **[SECURITY_FIXES_APPLIED.md](SECURITY_FIXES_APPLIED.md)** - Security updates
- **[COMMERCIAL_LICENSING.md](COMMERCIAL_LICENSING.md)** - Commercial licensing information

## 🔒 Security

LivingArchive includes enterprise-grade security features:

- ✅ API key authentication
- ✅ CORS protection
- ✅ Rate limiting
- ✅ Input validation
- ✅ SQL injection protection
- ✅ Secure defaults for production

**Security Checklist for Production:**
- [ ] Set `ENVIRONMENT=production`
- [ ] Configure strong `EGOLLAMA_API_KEY`
- [ ] Set `EGOLLAMA_REQUIRE_API_KEY=true`
- [ ] Restrict `EGOLLAMA_CORS_ORIGINS`
- [ ] Keep `EGOLLAMA_TRUST_REMOTE_CODE=false`
- [ ] Enable SSL/TLS (via reverse proxy)
- [ ] Review `SECURITY_FIXES_APPLIED.md`

For detailed security information, see [SECURITY_AUDIT_REPORT.md](SECURITY_AUDIT_REPORT.md).

## 🧪 Testing

Run the deployment test script:

```bash
./test_deployment.sh
```

This will test:
- Health endpoints
- API authentication
- Model loading
- Text generation
- Chat completions

## 🐛 Troubleshooting

### Common Issues

**Gateway won't start:**
```bash
# Check logs
docker-compose logs gateway  # Docker
tail -f logs/*.log           # Standalone

# Check ports
sudo lsof -i :8082
```

**Model not found:**
```bash
# List available models
curl http://localhost:8082/api/models

# Check Ollama is running
curl http://localhost:11434/api/tags
```

**Authentication errors:**
```bash
# Verify API key is set
echo $EGOLLAMA_API_KEY

# Test with API key
curl -H "X-API-Key: your-key" http://localhost:8082/health
```

For more troubleshooting tips, see [WIKI.md](WIKI.md#troubleshooting).

## 🤝 Contributing

Contributions are welcome! Please note:

1. **Contributor License Agreement (CLA):** All contributors must agree to the terms in [CONTRIBUTOR_LICENSE_AGREEMENT.md](CONTRIBUTOR_LICENSE_AGREEMENT.md)
2. **Code Style:** Follow Python PEP 8 guidelines
3. **Testing:** Ensure all tests pass before submitting
4. **Documentation:** Update relevant documentation for new features

## 📄 License

This project is dual-licensed:

- **Non-Commercial Use:** Apache License 2.0 (for personal, educational, non-profit use)
- **Commercial Use:** Business Source License 1.1 (commercial license required)

See [LICENSE](LICENSE) for details and [COMMERCIAL_LICENSING.md](COMMERCIAL_LICENSING.md) for commercial licensing information.

**Note:** For production or commercial deployments, a commercial license agreement is required. Contact: charlesmcgowen@gmail.com

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Model support via [Ollama](https://ollama.ai/) and [HuggingFace](https://huggingface.co/)
- Thanks to all contributors and the open-source community

## 📞 Support

- 📖 [Documentation](WIKI.md)
- 🐛 [Report Issues](https://github.com/CharlesMcGowen/LivingArchive/issues)
- 💬 [Discussions](https://github.com/CharlesMcGowen/LivingArchive/discussions)
- 📧 Email: charlesmcgowen@gmail.com

## 🗺️ Roadmap

- [ ] Additional model backend integrations
- [ ] Enhanced monitoring and observability
- [ ] Advanced caching strategies
- [ ] Multi-GPU support
- [ ] Load balancing across instances
- [ ] GraphQL API support

---

**Made with ❤️ for the LLM community**

