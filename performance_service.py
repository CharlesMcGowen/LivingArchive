#!/usr/bin/env python3
"""
Performance Optimization Service for LLaMA Gateway
==================================================

Provides monitoring, caching, and optimization for the gateway.
Tracks performance metrics and optimizes resource usage.

Author: EGO Revolution Team
Version: 1.0.0 - Phase 3 Advanced Features
"""

import asyncio
import logging
import json
import time
import psutil
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add GPU_Accelerator to path
sys.path.insert(0, str(Path(__file__).parent.parent / "GPU_Accelerator"))

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor system and application performance"""
    
    def __init__(self):
        self.metrics = {
            "requests": [],
            "response_times": [],
            "memory_usage": [],
            "cpu_usage": [],
            "gpu_usage": [],
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": []
        }
        
        self.start_time = time.time()
        self.gpu_available = False
        
        # Try to initialize GPU monitoring
        try:
            from accelerators.gpu_acceleration_service import GPUAccelerationService
            self.gpu_service = GPUAccelerationService()
            self.gpu_available = self.gpu_service.gpu_info.available if self.gpu_service.gpu_info else False
        except Exception as e:
            logger.warning(f"⚠️ GPU monitoring not available: {e}")
            self.gpu_service = None
    
    async def record_request(self, endpoint: str, response_time: float, status_code: int, 
                           memory_used: float = 0, gpu_used: float = 0):
        """Record a request and its metrics"""
        try:
            request_data = {
                "timestamp": datetime.now().isoformat(),
                "endpoint": endpoint,
                "response_time": response_time,
                "status_code": status_code,
                "memory_used": memory_used,
                "gpu_used": gpu_used
            }
            
            self.metrics["requests"].append(request_data)
            self.metrics["response_times"].append(response_time)
            
            # Keep only last 1000 requests
            if len(self.metrics["requests"]) > 1000:
                self.metrics["requests"] = self.metrics["requests"][-1000:]
                self.metrics["response_times"] = self.metrics["response_times"][-1000:]
            
            # Record system metrics
            await self._record_system_metrics()
            
        except Exception as e:
            logger.error(f"❌ Error recording request: {e}")
    
    async def _record_system_metrics(self):
        """Record system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.metrics["cpu_usage"].append({
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu_percent
            })
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics["memory_usage"].append({
                "timestamp": datetime.now().isoformat(),
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_available_gb": memory.available / (1024**3)
            })
            
            # GPU usage (if available)
            if self.gpu_available and self.gpu_service:
                try:
                    gpu_info = self.gpu_service.gpu_info
                    self.metrics["gpu_usage"].append({
                        "timestamp": datetime.now().isoformat(),
                        "gpu_utilization": 50,  # Simulated
                        "vram_used_gb": gpu_info.memory_gb * 0.3,  # Simulated
                        "vram_available_gb": gpu_info.memory_gb * 0.7
                    })
                except Exception as e:
                    logger.debug(f"GPU metrics not available: {e}")
            
            # Keep only last 100 metrics
            for key in ["cpu_usage", "memory_usage", "gpu_usage"]:
                if len(self.metrics[key]) > 100:
                    self.metrics[key] = self.metrics[key][-100:]
                    
        except Exception as e:
            logger.error(f"❌ Error recording system metrics: {e}")
    
    async def record_cache_hit(self):
        """Record a cache hit"""
        self.metrics["cache_hits"] += 1
    
    async def record_cache_miss(self):
        """Record a cache miss"""
        self.metrics["cache_misses"] += 1
    
    async def record_error(self, error_type: str, error_message: str, endpoint: str = None):
        """Record an error"""
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "endpoint": endpoint
        }
        self.metrics["errors"].append(error_data)
        
        # Keep only last 100 errors
        if len(self.metrics["errors"]) > 100:
            self.metrics["errors"] = self.metrics["errors"][-100:]
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            uptime = time.time() - self.start_time
            
            # Calculate averages
            avg_response_time = 0
            if self.metrics["response_times"]:
                avg_response_time = sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
            
            # Cache hit rate
            total_cache_requests = self.metrics["cache_hits"] + self.metrics["cache_misses"]
            cache_hit_rate = 0
            if total_cache_requests > 0:
                cache_hit_rate = self.metrics["cache_hits"] / total_cache_requests
            
            # Recent performance (last 10 minutes)
            recent_cutoff = datetime.now() - timedelta(minutes=10)
            recent_requests = [
                req for req in self.metrics["requests"]
                if datetime.fromisoformat(req["timestamp"]) > recent_cutoff
            ]
            
            recent_avg_response_time = 0
            if recent_requests:
                recent_avg_response_time = sum(req["response_time"] for req in recent_requests) / len(recent_requests)
            
            return {
                "uptime_seconds": uptime,
                "total_requests": len(self.metrics["requests"]),
                "average_response_time_ms": round(avg_response_time * 1000, 2),
                "recent_avg_response_time_ms": round(recent_avg_response_time * 1000, 2),
                "cache_hit_rate": round(cache_hit_rate * 100, 2),
                "cache_hits": self.metrics["cache_hits"],
                "cache_misses": self.metrics["cache_misses"],
                "total_errors": len(self.metrics["errors"]),
                "gpu_available": self.gpu_available,
                "system_metrics": {
                    "cpu_usage": self.metrics["cpu_usage"][-1] if self.metrics["cpu_usage"] else None,
                    "memory_usage": self.metrics["memory_usage"][-1] if self.metrics["memory_usage"] else None,
                    "gpu_usage": self.metrics["gpu_usage"][-1] if self.metrics["gpu_usage"] else None
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting performance stats: {e}")
            return {"error": str(e)}

class CacheService:
    """Service for caching responses and data"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if time.time() - entry["timestamp"] > self.ttl_seconds:
                    del self.cache[key]
                    self.misses += 1
                    return None
                
                self.hits += 1
                return entry["value"]
            else:
                self.misses += 1
                return None
                
        except Exception as e:
            logger.error(f"❌ Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any) -> bool:
        """Set value in cache"""
        try:
            # Remove oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
                del self.cache[oldest_key]
            
            self.cache[key] = {
                "value": value,
                "timestamp": time.time()
            }
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Cache delete error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache"""
        try:
            self.cache.clear()
            return True
        except Exception as e:
            logger.error(f"❌ Cache clear error: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "cache_size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate_percent": round(hit_rate, 2),
                "ttl_seconds": self.ttl_seconds
            }
        except Exception as e:
            logger.error(f"❌ Cache stats error: {e}")
            return {"error": str(e)}

class OptimizationService:
    """Service for performance optimization"""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.cache_service = CacheService()
        self.optimization_rules = []
        self._setup_optimization_rules()
    
    def _setup_optimization_rules(self):
        """Setup optimization rules"""
        self.optimization_rules = [
            {
                "name": "response_time_optimization",
                "condition": lambda stats: stats.get("average_response_time_ms", 0) > 1000,
                "action": "Enable caching for slow endpoints"
            },
            {
                "name": "memory_optimization",
                "condition": lambda stats: stats.get("system_metrics", {}).get("memory_usage", {}).get("memory_percent", 0) > 80,
                "action": "Clear cache and optimize memory usage"
            },
            {
                "name": "cache_optimization",
                "condition": lambda stats: stats.get("cache_hit_rate", 0) < 50,
                "action": "Increase cache TTL and size"
            }
        ]
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Run performance optimization"""
        try:
            stats = await self.performance_monitor.get_performance_stats()
            optimizations_applied = []
            
            for rule in self.optimization_rules:
                if rule["condition"](stats):
                    optimizations_applied.append({
                        "rule": rule["name"],
                        "action": rule["action"],
                        "applied": True
                    })
            
            # Apply specific optimizations
            if any(opt["rule"] == "response_time_optimization" for opt in optimizations_applied):
                # Enable more aggressive caching
                self.cache_service.ttl_seconds = 7200  # 2 hours
                self.cache_service.max_size = 2000
            
            if any(opt["rule"] == "memory_optimization" for opt in optimizations_applied):
                # Clear cache and optimize memory
                await self.cache_service.clear()
            
            if any(opt["rule"] == "cache_optimization" for opt in optimizations_applied):
                # Increase cache size
                self.cache_service.max_size = 3000
            
            return {
                "optimizations_applied": optimizations_applied,
                "total_optimizations": len(optimizations_applied),
                "performance_stats": stats
            }
            
        except Exception as e:
            logger.error(f"❌ Performance optimization error: {e}")
            return {"error": str(e)}
    
    async def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get optimization recommendations"""
        try:
            stats = await self.performance_monitor.get_performance_stats()
            recommendations = []
            
            # Response time recommendations
            if stats.get("average_response_time_ms", 0) > 500:
                recommendations.append({
                    "type": "response_time",
                    "priority": "high",
                    "recommendation": "Consider enabling response caching for frequently accessed endpoints",
                    "impact": "Reduce response time by 30-50%"
                })
            
            # Memory recommendations
            memory_usage = stats.get("system_metrics", {}).get("memory_usage", {}).get("memory_percent", 0)
            if memory_usage > 70:
                recommendations.append({
                    "type": "memory",
                    "priority": "medium",
                    "recommendation": "Consider clearing cache or reducing model memory usage",
                    "impact": "Free up 20-30% memory"
                })
            
            # Cache recommendations
            if stats.get("cache_hit_rate", 0) < 60:
                recommendations.append({
                    "type": "caching",
                    "priority": "medium",
                    "recommendation": "Increase cache size and TTL for better hit rates",
                    "impact": "Improve cache hit rate by 20-40%"
                })
            
            return {
                "recommendations": recommendations,
                "total_recommendations": len(recommendations),
                "high_priority": len([r for r in recommendations if r["priority"] == "high"]),
                "medium_priority": len([r for r in recommendations if r["priority"] == "medium"])
            }
            
        except Exception as e:
            logger.error(f"❌ Optimization recommendations error: {e}")
            return {"error": str(e)}

# Global services
performance_monitor = PerformanceMonitor()
cache_service = CacheService()
optimization_service = OptimizationService()

# Convenience functions
async def record_request(endpoint: str, response_time: float, status_code: int, 
                        memory_used: float = 0, gpu_used: float = 0):
    """Record a request"""
    await performance_monitor.record_request(endpoint, response_time, status_code, memory_used, gpu_used)

async def get_performance_stats() -> Dict[str, Any]:
    """Get performance statistics"""
    return await performance_monitor.get_performance_stats()

async def cache_get(key: str) -> Optional[Any]:
    """Get from cache"""
    return await cache_service.get(key)

async def cache_set(key: str, value: Any) -> bool:
    """Set in cache"""
    return await cache_service.set(key, value)

async def optimize_performance() -> Dict[str, Any]:
    """Optimize performance"""
    return await optimization_service.optimize_performance()

async def get_optimization_recommendations() -> Dict[str, Any]:
    """Get optimization recommendations"""
    return await optimization_service.get_optimization_recommendations()
