#!/usr/bin/env python3
"""
Service Watchdog - Auto-Recovery System
========================================

Monitors and automatically restarts services to prevent crashes:
- LLaMA Gateway monitoring
- EgoQT application monitoring
- Docker container health checks
- System resource monitoring

Author: L (Python Expert)
Version: 1.0.0 - Auto-Recovery Watchdog
"""

import subprocess
import time
import logging
import psutil
import signal
import sys
import json
import requests
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServiceWatchdog:
    """Monitors and restarts services automatically"""
    
    def __init__(self):
        self.services = {
            'llama_gateway': {
                'command': 'python3 /mnt/webapps-nvme/EgoLlama/standalone_llama_gateway.py',
                'working_dir': '/mnt/webapps-nvme/EgoLlama',
                'port': 8082,
                'max_memory_mb': 2000,
                'health_endpoint': '/health',
                'restart_count': 0,
                'last_restart': None,
                'max_restarts_per_hour': 3
            },
            'egoqt': {
                'command': './run.sh',
                'working_dir': '/mnt/webapps-nvme/EgoQT',
                'port': None,  # No specific port
                'max_memory_mb': 4000,
                'health_endpoint': None,
                'restart_count': 0,
                'last_restart': None,
                'max_restarts_per_hour': 2
            }
        }
        
        self.monitoring = False
        self.check_interval = 30  # Check every 30 seconds
        self.log_file = Path('/mnt/webapps-nvme/EgoLlama/watchdog.log')
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"🛑 Received signal {signum}, shutting down watchdog...")
        self.monitoring = False
        sys.exit(0)
    
    def log_event(self, service_name: str, event: str, details: str = ""):
        """Log events to file"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'service': service_name,
            'event': event,
            'details': details
        }
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")
    
    def check_service_health(self, service_name: str, config: Dict) -> bool:
        """Check if service is healthy"""
        try:
            # Check if process is running
            process_found = False
            memory_usage = 0
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
                try:
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    
                    # Check for LLaMA Gateway
                    if service_name == 'llama_gateway' and 'simple_llama_gateway_crash_safe.py' in cmdline:
                        process_found = True
                        memory_usage = proc.info['memory_info'].rss / 1024 / 1024
                        break
                    
                    # Check for EgoQT
                    elif service_name == 'egoqt' and 'python main.py' in cmdline and 'EgoQT' in cmdline:
                        process_found = True
                        memory_usage = proc.info['memory_info'].rss / 1024 / 1024
                        break
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if not process_found:
                logger.warning(f"⚠️ {service_name} process not found")
                self.log_event(service_name, "process_not_found", "Service process not running")
                return False
            
            # Check memory usage
            if memory_usage > config['max_memory_mb']:
                logger.warning(f"⚠️ {service_name} using too much memory: {memory_usage:.1f}MB")
                self.log_event(service_name, "high_memory", f"Memory usage: {memory_usage:.1f}MB")
                return False
            
            # Check HTTP health endpoint if available
            if config.get('health_endpoint') and config.get('port'):
                try:
                    response = requests.get(
                        f"http://localhost:{config['port']}{config['health_endpoint']}",
                        timeout=5
                    )
                    if response.status_code != 200:
                        logger.warning(f"⚠️ {service_name} health check failed: {response.status_code}")
                        self.log_event(service_name, "health_check_failed", f"Status: {response.status_code}")
                        return False
                except requests.exceptions.RequestException as e:
                    logger.warning(f"⚠️ {service_name} health check error: {e}")
                    self.log_event(service_name, "health_check_error", str(e))
                    return False
            
            logger.debug(f"✅ {service_name} is healthy (memory: {memory_usage:.1f}MB)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking {service_name} health: {e}")
            self.log_event(service_name, "health_check_error", str(e))
            return False
    
    def can_restart(self, service_name: str, config: Dict) -> bool:
        """Check if service can be restarted (rate limiting)"""
        now = datetime.now()
        
        # Reset restart count if it's been more than an hour
        if config['last_restart']:
            time_since_restart = (now - config['last_restart']).total_seconds()
            if time_since_restart > 3600:  # 1 hour
                config['restart_count'] = 0
        
        return config['restart_count'] < config['max_restarts_per_hour']
    
    def restart_service(self, service_name: str, config: Dict):
        """Restart a service"""
        if not self.can_restart(service_name, config):
            logger.error(f"❌ {service_name} restart rate limited ({config['restart_count']}/{config['max_restarts_per_hour']} per hour)")
            self.log_event(service_name, "restart_rate_limited", f"Count: {config['restart_count']}")
            return False
        
        logger.info(f"🔄 Restarting {service_name}...")
        self.log_event(service_name, "restart_initiated", f"Restart #{config['restart_count'] + 1}")
        
        try:
            # Kill existing processes
            self._kill_service_processes(service_name)
            
            # Wait a moment
            time.sleep(2)
            
            # Create log file for this restart
            restart_log = f"/tmp/{service_name}_restart_{int(time.time())}.log"
            
            # Start new process
            if config.get('working_dir'):
                with open(restart_log, 'w') as log_file:
                    process = subprocess.Popen(
                        config['command'],
                        shell=True,
                        cwd=config['working_dir'],
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        preexec_fn=os.setsid if hasattr(os, 'setsid') else None
                    )
            else:
                with open(restart_log, 'w') as log_file:
                    process = subprocess.Popen(
                        config['command'],
                        shell=True,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        preexec_fn=os.setsid if hasattr(os, 'setsid') else None
                    )
            
            # Update restart tracking
            config['restart_count'] += 1
            config['last_restart'] = datetime.now()
            
            logger.info(f"✅ {service_name} restart initiated (PID: {process.pid})")
            self.log_event(service_name, "restart_completed", f"PID: {process.pid}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to restart {service_name}: {e}")
            self.log_event(service_name, "restart_failed", str(e))
            return False
    
    def _kill_service_processes(self, service_name: str):
        """Kill existing service processes"""
        try:
            killed_count = 0
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    
                    # Kill LLaMA Gateway processes
                    if service_name == 'llama_gateway' and 'simple_llama_gateway' in cmdline:
                        proc.terminate()
                        killed_count += 1
                    
                    # Kill EgoQT processes
                    elif service_name == 'egoqt' and 'python main.py' in cmdline and 'EgoQT' in cmdline:
                        proc.terminate()
                        killed_count += 1
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if killed_count > 0:
                logger.info(f"🔪 Killed {killed_count} {service_name} processes")
                time.sleep(1)  # Give processes time to terminate
                
        except Exception as e:
            logger.error(f"❌ Error killing {service_name} processes: {e}")
    
    def check_docker_containers(self):
        """Check Docker container health"""
        try:
            result = subprocess.run(['docker', 'ps', '--format', 'json'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                logger.warning("⚠️ Docker not available or error")
                return
            
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    try:
                        containers.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            
            unhealthy_containers = [c for c in containers if c.get('Status', '').find('unhealthy') != -1]
            
            if unhealthy_containers:
                logger.warning(f"⚠️ Found {len(unhealthy_containers)} unhealthy Docker containers")
                for container in unhealthy_containers:
                    logger.warning(f"  - {container.get('Names', 'Unknown')}: {container.get('Status', 'Unknown')}")
                    self.log_event("docker", "unhealthy_container", 
                                 f"Container: {container.get('Names', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"❌ Error checking Docker containers: {e}")
    
    def check_system_resources(self):
        """Check overall system resource usage"""
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('/')
            
            # Log warnings for high resource usage
            if memory.percent > 90:
                logger.warning(f"⚠️ High memory usage: {memory.percent:.1f}%")
                self.log_event("system", "high_memory", f"Usage: {memory.percent:.1f}%")
            
            if cpu > 90:
                logger.warning(f"⚠️ High CPU usage: {cpu:.1f}%")
                self.log_event("system", "high_cpu", f"Usage: {cpu:.1f}%")
            
            if disk.percent > 90:
                logger.warning(f"⚠️ High disk usage: {disk.percent:.1f}%")
                self.log_event("system", "high_disk", f"Usage: {disk.percent:.1f}%")
            
            logger.debug(f"📊 System resources - Memory: {memory.percent:.1f}%, CPU: {cpu:.1f}%, Disk: {disk.percent:.1f}%")
            
        except Exception as e:
            logger.error(f"❌ Error checking system resources: {e}")
    
    def run(self):
        """Main watchdog loop"""
        logger.info("🐕 Service Watchdog started")
        logger.info(f"📁 Log file: {self.log_file}")
        logger.info(f"⏰ Check interval: {self.check_interval} seconds")
        logger.info("🔍 Monitoring services:")
        for service_name, config in self.services.items():
            logger.info(f"  - {service_name}: {config['command']}")
        
        self.monitoring = True
        self.log_event("watchdog", "started", f"Monitoring {len(self.services)} services")
        
        while self.monitoring:
            try:
                # Check each service
                for service_name, config in self.services.items():
                    if not self.check_service_health(service_name, config):
                        if self.can_restart(service_name, config):
                            self.restart_service(service_name, config)
                        else:
                            logger.error(f"❌ {service_name} is unhealthy but restart rate limited")
                
                # Check Docker containers
                self.check_docker_containers()
                
                # Check system resources
                self.check_system_resources()
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Watchdog stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Watchdog error: {e}")
                self.log_event("watchdog", "error", str(e))
                time.sleep(self.check_interval)
        
        self.log_event("watchdog", "stopped", "Watchdog service stopped")
        logger.info("🐕 Service Watchdog stopped")

def main():
    """Start the watchdog service"""
    print("=" * 80)
    print("Service Watchdog - Auto-Recovery System")
    print("=" * 80)
    print()
    print("Features:")
    print("  ✅ LLaMA Gateway monitoring")
    print("  ✅ EgoQT application monitoring")
    print("  ✅ Docker container health checks")
    print("  ✅ System resource monitoring")
    print("  ✅ Automatic service restart")
    print("  ✅ Rate limiting protection")
    print()
    print("=" * 80)
    print()
    
    watchdog = ServiceWatchdog()
    watchdog.run()

if __name__ == "__main__":
    main()
