#!/bin/bash
# ==============================================================================
# EgoLlama-clean Quick Start Script
# ==============================================================================
# One-command deployment for EgoLlama Gateway
# Supports both Docker and standalone modes
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print colored messages
info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

success() {
    echo -e "${GREEN}✅${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

error() {
    echo -e "${RED}❌${NC} $1"
}

title() {
    echo -e "${CYAN}$1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect deployment mode
detect_mode() {
    # Check for docker compose v2 first (preferred)
    if command_exists docker && docker compose version >/dev/null 2>&1; then
        if docker ps >/dev/null 2>&1; then
            echo "docker"
            return 0
        fi
    fi
    
    # Fallback to docker-compose v1
    if command_exists docker && command_exists docker-compose; then
        if docker ps >/dev/null 2>&1; then
            echo "docker"
            return 0
        fi
    fi
    
    if command_exists python3; then
        echo "standalone"
        return 0
    fi
    
    echo "none"
    return 1
}

# Docker deployment
deploy_docker() {
    title "🐳 Deploying with Docker..."
    
    # Check prerequisites
    if ! command_exists docker; then
        error "Docker is not installed"
        echo "  Install: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check for docker compose v2 or docker-compose v1
    if ! docker compose version >/dev/null 2>&1 && ! command_exists docker-compose; then
        error "Docker Compose is not installed"
        echo "  Install: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    # Determine which compose command to use (use v2 if available, fallback to v1)
    local COMPOSE_CMD
    if docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
        info "Using Docker Compose v2"
    else
        COMPOSE_CMD="docker-compose"
        info "Using Docker Compose v1"
    fi
    
    # Setup environment
    if [ ! -f .env ]; then
        if [ -f env.example ]; then
            cp env.example .env
            success "Created .env from env.example"
            info "💡 Edit .env to configure security settings (API keys, CORS, etc.)"
        else
            warning "env.example not found, creating basic .env"
            cat > .env << EOF
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=ego
GATEWAY_PORT=8082
# Security settings (recommended for production)
# EGOLLAMA_API_KEY=your-secret-api-key-here
# EGOLLAMA_REQUIRE_API_KEY=false
# EGOLLAMA_CORS_ORIGINS=http://localhost:9000,http://localhost:3000
# EGOLLAMA_HOST=0.0.0.0
EOF
        fi
    else
        success ".env already exists"
    fi
    
    # Start services
    info "Starting Docker services..."
    $COMPOSE_CMD up -d
    
    success "Services started!"
    info "Waiting for services to be healthy..."
    
    # Wait for services
    local max_attempts=30
    local attempt=0
    
    # PostgreSQL
    while [ $attempt -lt $max_attempts ]; do
        if $COMPOSE_CMD exec -T postgres pg_isready -U postgres >/dev/null 2>&1; then
            success "PostgreSQL is ready"
            # Run database migrations
            info "Running database migrations..."
            if $COMPOSE_CMD exec -T gateway alembic upgrade head >/dev/null 2>&1; then
                success "Database migrations completed"
            else
                warning "Migration failed (may already be up to date)"
            fi
            break
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    # Redis
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if $COMPOSE_CMD exec -T redis redis-cli ping >/dev/null 2>&1; then
            success "Redis is ready"
            break
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    # Gateway
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -f http://localhost:8082/health >/dev/null 2>&1; then
            success "Gateway is ready"
            break
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        warning "Gateway took longer than expected"
        info "Check logs: $COMPOSE_CMD logs gateway"
    fi
}

# Standalone deployment
deploy_standalone() {
    title "🐍 Deploying Standalone (Python)..."
    
    # Check Python
    if ! command_exists python3; then
        error "Python 3 is not installed"
        echo "  Install Python 3.11+ and try again"
        exit 1
    fi
    
    local python_version=$(python3 --version 2>&1 | awk '{print $2}')
    info "Python version: $python_version"
    
    # Check virtual environment
    if [ ! -d "venv" ]; then
        info "Creating virtual environment..."
        python3 -m venv venv
        success "Virtual environment created"
    fi
    
    # Activate virtual environment
    info "Activating virtual environment..."
    source venv/bin/activate
    
    # Install dependencies
    info "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    success "Dependencies installed"
    
    # Setup environment variables
    if [ ! -f .env ]; then
        warning ".env not found, using defaults"
        info "💡 Create .env file for production configuration"
    fi
    
    # Check if PostgreSQL and Redis are available
    info "Checking dependencies..."
    
    local missing=0
    
    # Check PostgreSQL (optional)
    if command_exists psql; then
        success "PostgreSQL client found"
    else
        warning "PostgreSQL client not found (optional)"
    fi
    
    # Check Redis (optional)
    if command_exists redis-cli; then
        success "Redis client found"
    else
        warning "Redis client not found (optional)"
    fi
    
    info "💡 Gateway will work without PostgreSQL/Redis (graceful degradation)"
}

# Show status
show_status() {
    echo ""
    title "============================================================================"
    success "Deployment Complete!"
    title "============================================================================"
    echo ""
    
    local mode=$(detect_mode)
    
    if [ "$mode" = "docker" ]; then
        # Determine compose command
        if docker compose version >/dev/null 2>&1; then
            COMPOSE_CMD="docker compose"
        else
            COMPOSE_CMD="docker-compose"
        fi
        
        echo "Services:"
        $COMPOSE_CMD ps
        echo ""
        echo "Gateway URL: http://localhost:8082"
        echo "Health Check: http://localhost:8082/health"
        echo ""
        echo "Useful commands:"
        echo "  View logs:    $COMPOSE_CMD logs -f gateway"
        echo "  Stop:         $COMPOSE_CMD down"
        echo "  Restart:      $COMPOSE_CMD restart"
        echo "  Status:       $COMPOSE_CMD ps"
    else
        echo "Gateway URL: http://localhost:8082"
        echo "Health Check: http://localhost:8082/health"
        echo ""
        echo "To start the gateway:"
        echo "  source venv/bin/activate"
        echo "  python simple_llama_gateway_crash_safe.py"
        echo ""
        echo "Or use the startup script:"
        echo "  ./start_v2_gateway.sh"
    fi
    
    echo ""
    echo "📚 Documentation:"
    echo "  README.md - Full documentation"
    echo "  SETUP_GUIDE.md - Detailed setup guide"
    echo "  SECURITY_FIXES_APPLIED.md - Security configuration"
    echo ""
    echo "🔒 Security:"
    echo "  Configure API keys and CORS in .env file"
    echo "  See SECURITY_FIXES_APPLIED.md for details"
    echo ""
}

# Test deployment
test_deployment() {
    info "Testing gateway..."
    
    local max_attempts=10
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -f http://localhost:8082/health >/dev/null 2>&1; then
            success "Gateway health check passed"
            
            local health=$(curl -s http://localhost:8082/health)
            echo "  Health status: $health"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    warning "Gateway health check failed"
    info "Gateway may still be starting up..."
    return 1
}

# Main execution
main() {
    echo ""
    title "============================================================================"
    title "  EgoLlama-clean Gateway - Quick Start"
    title "============================================================================"
    echo ""
    
    # Detect mode
    local mode=$(detect_mode)
    
    if [ "$mode" = "none" ]; then
        error "No deployment method available"
        echo ""
        echo "Please install either:"
        echo "  1. Docker + Docker Compose (recommended)"
        echo "  2. Python 3.11+ (for standalone)"
        exit 1
    fi
    
    info "Detected deployment mode: $mode"
    echo ""
    
    # Deploy based on mode
    if [ "$mode" = "docker" ]; then
        deploy_docker
    else
        deploy_standalone
    fi
    
    echo ""
    
    # Test deployment
    test_deployment
    
    echo ""
    
    # Show status
    show_status
}

# Run main function
main

