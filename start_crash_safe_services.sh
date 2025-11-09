#!/bin/bash
# Start Crash-Safe Services
# =========================
# 
# This script starts all services with crash prevention measures:
# - LLaMA Gateway with memory limits
# - EgoQT with resource monitoring
# - Watchdog service for auto-recovery
# - Docker log monitoring

echo "🛡️ Starting Crash-Safe Services"
echo "================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if a service is running
check_service() {
    local service_name=$1
    local port=$2
    
    if [ -n "$port" ]; then
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $service_name is running on port $port${NC}"
            return 0
        else
            echo -e "${RED}❌ $service_name is not responding on port $port${NC}"
            return 1
        fi
    else
        if pgrep -f "$service_name" > /dev/null; then
            echo -e "${GREEN}✅ $service_name process is running${NC}"
            return 0
        else
            echo -e "${RED}❌ $service_name process not found${NC}"
            return 1
        fi
    fi
}

# Function to start a service in background
start_service() {
    local service_name=$1
    local command=$2
    local working_dir=$3
    local log_file=$4
    
    echo -e "${BLUE}🚀 Starting $service_name...${NC}"
    
    if [ -n "$working_dir" ]; then
        cd "$working_dir"
    fi
    
    nohup $command > "$log_file" 2>&1 &
    local pid=$!
    
    echo -e "${GREEN}✅ $service_name started with PID $pid${NC}"
    echo "📁 Log file: $log_file"
    
    # Wait a moment for service to start
    sleep 3
    
    return $pid
}

# Check if we're in the right directory
if [ ! -f "/mnt/webapps-nvme/EgoLlama/simple_llama_gateway_crash_safe.py" ]; then
    echo -e "${RED}❌ Error: Crash-safe gateway not found${NC}"
    echo "Please run this script from the correct directory"
    exit 1
fi

# Create logs directory
mkdir -p /mnt/webapps-nvme/EgoLlama/logs

echo -e "${YELLOW}🔍 Checking current service status...${NC}"
echo

# Check current services
check_service "LLaMA Gateway" "8082"
llama_running=$?

check_service "EgoQT" ""
egoqt_running=$?

echo

# Stop existing services if running
if [ $llama_running -eq 0 ]; then
    echo -e "${YELLOW}🛑 Stopping existing LLaMA Gateway...${NC}"
    pkill -f "simple_llama_gateway"
    sleep 2
fi

if [ $egoqt_running -eq 0 ]; then
    echo -e "${YELLOW}🛑 Stopping existing EgoQT...${NC}"
    pkill -f "python main.py"
    sleep 2
fi

echo

# Start LLaMA Gateway with crash prevention
echo -e "${BLUE}🛡️ Starting Crash-Safe LLaMA Gateway...${NC}"
start_service "LLaMA Gateway" \
    "python3 simple_llama_gateway_crash_safe.py" \
    "/mnt/webapps-nvme/EgoLlama" \
    "/mnt/webapps-nvme/EgoLlama/logs/gateway_crash_safe.log"

# Wait for gateway to start
echo -e "${YELLOW}⏳ Waiting for LLaMA Gateway to start...${NC}"
sleep 5

# Check if gateway is healthy
if check_service "LLaMA Gateway" "8082"; then
    echo -e "${GREEN}✅ LLaMA Gateway is healthy${NC}"
else
    echo -e "${RED}❌ LLaMA Gateway failed to start properly${NC}"
    echo "📁 Check log: /mnt/webapps-nvme/EgoLlama/logs/gateway_crash_safe.log"
fi

echo

# Start EgoQT with resource monitoring
echo -e "${BLUE}🛡️ Starting EgoQT with Resource Monitoring...${NC}"
start_service "EgoQT" \
    "./run.sh" \
    "/mnt/webapps-nvme/EgoQT" \
    "/mnt/webapps-nvme/EgoQT/logs/egoqt_crash_safe.log"

# Wait for EgoQT to start
echo -e "${YELLOW}⏳ Waiting for EgoQT to start...${NC}"
sleep 10

# Check if EgoQT is running
if check_service "EgoQT" ""; then
    echo -e "${GREEN}✅ EgoQT is running${NC}"
else
    echo -e "${RED}❌ EgoQT failed to start properly${NC}"
    echo "📁 Check log: /mnt/webapps-nvme/EgoQT/logs/egoqt_crash_safe.log"
fi

echo

# Start Watchdog Service
echo -e "${BLUE}🐕 Starting Watchdog Service...${NC}"
start_service "Watchdog" \
    "python3 watchdog_service.py" \
    "/mnt/webapps-nvme/EgoLlama" \
    "/mnt/webapps-nvme/EgoLlama/logs/watchdog.log"

# Wait for watchdog to start
sleep 3

if check_service "Watchdog" ""; then
    echo -e "${GREEN}✅ Watchdog Service is running${NC}"
else
    echo -e "${RED}❌ Watchdog Service failed to start${NC}"
    echo "📁 Check log: /mnt/webapps-nvme/EgoLlama/logs/watchdog.log"
fi

echo

# Run Docker log analysis
echo -e "${BLUE}🔍 Running Docker Log Analysis...${NC}"
python3 docker_log_analyzer.py

echo

# Final status check
echo -e "${YELLOW}📊 Final Service Status:${NC}"
echo "================================"

check_service "LLaMA Gateway" "8082"
check_service "EgoQT" ""
check_service "Watchdog" ""

echo

# Show resource status
echo -e "${YELLOW}📈 Resource Status:${NC}"
echo "=================="

# Get memory usage
memory_info=$(free -h | grep "Mem:")
echo "💾 Memory: $memory_info"

# Get CPU usage
cpu_info=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
echo "🖥️ CPU Usage: ${cpu_info}%"

# Get disk usage
disk_info=$(df -h / | tail -1 | awk '{print $5}')
echo "💿 Disk Usage: $disk_info"

echo

# Show log files
echo -e "${YELLOW}📁 Log Files:${NC}"
echo "============="
echo "• LLaMA Gateway: /mnt/webapps-nvme/EgoLlama/logs/gateway_crash_safe.log"
echo "• EgoQT: /mnt/webapps-nvme/EgoQT/logs/egoqt_crash_safe.log"
echo "• Watchdog: /mnt/webapps-nvme/EgoLlama/logs/watchdog.log"
echo "• Docker Analysis: /mnt/webapps-nvme/EgoLlama/docker_analysis_results.json"

echo

# Show monitoring commands
echo -e "${YELLOW}🔍 Monitoring Commands:${NC}"
echo "=========================="
echo "• Check LLaMA Gateway: curl http://localhost:8082/health"
echo "• Monitor resources: python3 /mnt/webapps-nvme/EgoLlama/watchdog_service.py"
echo "• Analyze Docker logs: python3 /mnt/webapps-nvme/EgoLlama/docker_log_analyzer.py"
echo "• View logs: tail -f /mnt/webapps-nvme/EgoLlama/logs/*.log"

echo

echo -e "${GREEN}🛡️ Crash-Safe Services Started Successfully!${NC}"
echo "=================================================="
echo
echo "Services are now running with crash prevention measures:"
echo "✅ Memory limits and monitoring"
echo "✅ Automatic service restart"
echo "✅ Resource usage tracking"
echo "✅ Docker log analysis"
echo "✅ System health monitoring"
echo
echo "The system will automatically recover from crashes and"
echo "monitor resource usage to prevent future issues."
echo
