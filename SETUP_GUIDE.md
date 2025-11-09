# EgoLlama Setup Guide - Difficulty Assessment

## Quick Difficulty Rating

| Setup Type | Difficulty | Time Required | Prerequisites |
|------------|-----------|----------------|---------------|
| **Docker (Recommended)** | ⭐ Easy | 5-10 minutes | Docker + Docker Compose |
| **Local Development** | ⭐⭐ Moderate | 15-30 minutes | Python 3.11+, PostgreSQL, Redis |
| **GPU Setup (NVIDIA)** | ⭐⭐ Moderate | 10-20 minutes | NVIDIA GPU + CUDA drivers |
| **GPU Setup (AMD)** | ⭐⭐⭐ Advanced | 20-40 minutes | AMD GPU + ROCm installation |

## Setup Methods

### Method 1: Docker (Easiest) ⭐

**Difficulty:** Easy  
**Time:** 5-10 minutes  
**Prerequisites:** Docker and Docker Compose installed

This is the recommended method for most users.

#### Step-by-Step:

1. **Check Prerequisites** (1 minute)
   ```bash
   docker --version
   docker-compose --version
   ```
   ✅ If both commands work, you're ready!

2. **Configure Environment** (1 minute)
   ```bash
   cp env.example .env
   # Edit .env if needed (defaults work for testing)
   ```

3. **Start Services** (2-3 minutes)
   ```bash
   docker-compose up -d
   ```
   This downloads images and starts PostgreSQL, Redis, and Gateway.

4. **Verify** (1 minute)
   ```bash
   curl http://localhost:8082/health
   ```

**Total Time:** ~5 minutes  
**Difficulty:** Easy - Just 3 commands!

---

### Method 2: Local Development ⭐⭐

**Difficulty:** Moderate  
**Time:** 15-30 minutes  
**Prerequisites:** Python 3.11+, PostgreSQL, Redis

#### Step-by-Step:

1. **Install Python** (5 minutes)
   ```bash
   python3 --version  # Should be 3.11+
   ```

2. **Install PostgreSQL** (10 minutes)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install postgresql postgresql-contrib
   
   # macOS
   brew install postgresql
   
   # Create database
   sudo -u postgres createdb ego
   ```

3. **Install Redis** (5 minutes)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install redis-server
   
   # macOS
   brew install redis
   ```

4. **Install Python Dependencies** (5 minutes)
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure Environment** (2 minutes)
   ```bash
   export EGOLLAMA_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/ego?sslmode=disable"
   export REDIS_HOST="localhost"
   export REDIS_PORT="6379"
   ```

6. **Run Gateway** (1 minute)
   ```bash
   python simple_llama_gateway_crash_safe.py
   ```

**Total Time:** ~30 minutes  
**Difficulty:** Moderate - Requires installing multiple services

---

### Method 3: GPU Setup (NVIDIA) ⭐⭐

**Difficulty:** Moderate  
**Time:** 10-20 minutes  
**Prerequisites:** NVIDIA GPU, CUDA drivers

#### Step-by-Step:

1. **Verify GPU** (1 minute)
   ```bash
   nvidia-smi
   ```
   ✅ Should show your GPU

2. **Install NVIDIA Docker Runtime** (10 minutes)
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

3. **Update docker-compose.yml** (2 minutes)
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

4. **Update requirements.txt** (1 minute)
   ```
   torch>=2.0.0+cu118
   ```

5. **Rebuild and Start** (5 minutes)
   ```bash
   docker-compose build --no-cache gateway
   docker-compose up -d
   ```

**Total Time:** ~20 minutes  
**Difficulty:** Moderate - Requires NVIDIA Docker setup

---

### Method 4: GPU Setup (AMD) ⭐⭐⭐

**Difficulty:** Advanced  
**Time:** 20-40 minutes  
**Prerequisites:** AMD GPU, ROCm installation

#### Step-by-Step:

1. **Install ROCm** (20-30 minutes)
   ```bash
   # Follow AMD ROCm installation guide
   # https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html
   ```

2. **Install ROCm Docker Runtime** (5 minutes)
   ```bash
   # Follow ROCm Docker documentation
   ```

3. **Update docker-compose.yml** (2 minutes)
   ```yaml
   gateway:
     deploy:
       resources:
         reservations:
           devices:
             - driver: rocm
   ```

4. **Update requirements.txt** (1 minute)
   ```
   torch>=2.0.0+rocm5.7
   ```

5. **Rebuild and Start** (5 minutes)
   ```bash
   docker-compose build --no-cache gateway
   docker-compose up -d
   ```

**Total Time:** ~40 minutes  
**Difficulty:** Advanced - ROCm installation can be complex

---

## Pre-Flight Checklist

Before starting, check these:

### Docker Method
- [ ] Docker installed (`docker --version`)
- [ ] Docker Compose installed (`docker-compose --version`)
- [ ] Docker daemon running (`docker ps`)
- [ ] Ports 5432, 6379, 8082 available

### Local Development
- [ ] Python 3.11+ installed
- [ ] PostgreSQL installed and running
- [ ] Redis installed and running
- [ ] `pip` available
- [ ] Database created (`ego`)

### GPU Setup
- [ ] GPU detected (`nvidia-smi` or ROCm tools)
- [ ] GPU drivers installed
- [ ] Docker GPU runtime configured

---

## Common Issues & Solutions

### Issue: "Port already in use"
**Difficulty:** Easy  
**Solution:**
```bash
# Check what's using the port
sudo lsof -i :8082
# Stop the conflicting service or change port in .env
```

### Issue: "Cannot connect to database"
**Difficulty:** Easy  
**Solution:**
```bash
# Check PostgreSQL is running
docker-compose ps postgres
# Check connection string in .env
```

### Issue: "Docker permission denied"
**Difficulty:** Easy  
**Solution:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

### Issue: "GPU not detected"
**Difficulty:** Moderate  
**Solution:**
- Verify GPU drivers: `nvidia-smi` or ROCm tools
- Check Docker GPU runtime: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
- Verify docker-compose.yml GPU configuration

---

## Automated Setup Script

For the easiest setup, use the automated script:

```bash
./setup.sh
```

This script:
1. ✅ Checks prerequisites
2. ✅ Creates .env from template
3. ✅ Starts Docker services
4. ✅ Waits for services to be healthy
5. ✅ Tests the gateway
6. ✅ Shows you the status

**Difficulty:** ⭐ Super Easy (just run one command!)

---

## Time Estimates

| User Experience Level | Docker Setup | Local Setup | GPU Setup |
|----------------------|--------------|-------------|-----------|
| **Beginner** | 10-15 min | 30-45 min | 30-60 min |
| **Intermediate** | 5-10 min | 15-30 min | 20-40 min |
| **Advanced** | 3-5 min | 10-15 min | 10-20 min |

---

## Recommendation

**For most users:** Use Docker method (⭐ Easy, 5-10 minutes)

**For developers:** Use Local Development method (⭐⭐ Moderate, 15-30 minutes)

**For GPU users:** Follow NVIDIA setup if available (⭐⭐ Moderate), AMD setup if needed (⭐⭐⭐ Advanced)

---

## Need Help?

1. Check the [README.md](README.md) for detailed instructions
2. Review [Troubleshooting](#common-issues--solutions) section
3. Check Docker logs: `docker-compose logs gateway`
4. Verify health: `curl http://localhost:8082/health`








