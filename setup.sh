#!/bin/bash
# ==============================================================================
# EgoLlama Automated Setup Script
# ==============================================================================
# This script automates the setup process for EgoLlama Gateway
# Difficulty: ⭐ Super Easy - Just run this script!
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    local missing=0
    
    if ! command_exists docker; then
        error "Docker is not installed"
        echo "  Install: https://docs.docker.com/get-docker/"
        missing=1
    else
        success "Docker found: $(docker --version)"
    fi
    
    if ! command_exists docker-compose; then
        error "Docker Compose is not installed"
        echo "  Install: https://docs.docker.com/compose/install/"
        missing=1
    else
        success "Docker Compose found: $(docker-compose --version)"
    fi
    
    if [ $missing -eq 1 ]; then
        error "Missing prerequisites. Please install them and try again."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker ps >/dev/null 2>&1; then
        error "Docker daemon is not running"
        echo "  Start Docker and try again"
        exit 1
    fi
    
    success "All prerequisites met!"
}

# Create .env file if it doesn't exist
setup_env() {
    info "Setting up environment..."
    
    if [ ! -f .env ]; then
        if [ -f env.example ]; then
            cp env.example .env
            success "Created .env from env.example"
            info "Using default configuration (you can edit .env later)"
        else
            warning "env.example not found, creating basic .env"
            cat > .env << EOF
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=ego
GATEWAY_PORT=8082
EOF
        fi
    else
        success ".env already exists, skipping"
    fi
}

# Check if ports are available
check_ports() {
    info "Checking if ports are available..."
    
    local ports=(5432 6379 8082)
    local ports_in_use=0
    
    for port in "${ports[@]}"; do
        if command_exists lsof; then
            if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
                warning "Port $port is already in use"
                ports_in_use=1
            else
                success "Port $port is available"
            fi
        elif command_exists netstat; then
            if netstat -tuln | grep -q ":$port "; then
                warning "Port $port is already in use"
                ports_in_use=1
            else
                success "Port $port is available"
            fi
        fi
    done
    
    if [ $ports_in_use -eq 1 ]; then
        warning "Some ports are in use. Services might fail to start."
        echo "  You can stop conflicting services or change ports in .env"
    fi
}

# Start Docker services
start_services() {
    info "Starting Docker services..."
    
    docker-compose up -d
    
    success "Services started!"
    info "Waiting for services to be healthy..."
    
    # Wait for PostgreSQL
    local max_attempts=30
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if docker-compose exec -T postgres pg_isready -U postgres >/dev/null 2>&1; then
            success "PostgreSQL is ready"
            break
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        warning "PostgreSQL took longer than expected to start"
    fi
    
    # Wait for Redis
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if docker-compose exec -T redis redis-cli ping >/dev/null 2>&1; then
            success "Redis is ready"
            break
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        warning "Redis took longer than expected to start"
    fi
    
    # Wait for Gateway
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
        warning "Gateway took longer than expected to start"
        info "Check logs with: docker-compose logs gateway"
    fi
}

# Test the gateway
test_gateway() {
    info "Testing gateway..."
    
    if curl -f http://localhost:8082/health >/dev/null 2>&1; then
        success "Gateway health check passed"
        
        # Get health status
        local health=$(curl -s http://localhost:8082/health)
        echo "  Health status: $health"
    else
        error "Gateway health check failed"
        info "Check logs with: docker-compose logs gateway"
        return 1
    fi
}

# Show status
show_status() {
    echo ""
    echo "============================================================================"
    success "Setup Complete!"
    echo "============================================================================"
    echo ""
    echo "Services:"
    docker-compose ps
    echo ""
    echo "Gateway URL: http://localhost:8082"
    echo "Health Check: http://localhost:8082/health"
    echo ""
    echo "Useful commands:"
    echo "  View logs:    docker-compose logs -f gateway"
    echo "  Stop services: docker-compose down"
    echo "  Restart:      docker-compose restart"
    echo ""
    echo "For more information, see README.md and SETUP_GUIDE.md"
    echo ""
}

# Main execution
main() {
    echo "============================================================================"
    echo "  EgoLlama Gateway - Automated Setup"
    echo "============================================================================"
    echo ""
    
    check_prerequisites
    echo ""
    
    setup_env
    echo ""
    
    check_ports
    echo ""
    
    start_services
    echo ""
    
    test_gateway
    echo ""
    
    show_status
}

# Run main function
main








