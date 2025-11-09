#!/usr/bin/env python3
"""
Redis Cache Module
==================

Redis-based caching, session management, rate limiting, and model storage for EgoLlama.
"""

import json
import logging
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

import redis
from redis.connection import ConnectionPool
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

from config import (
    REDIS_POOL_CONFIG,
    CACHE_KEY_PREFIX,
    CACHE_KEY_CACHE,
    CACHE_KEY_SESSION,
    CACHE_KEY_RATE_LIMIT,
    CACHE_KEY_MODEL,
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW,
    ENABLE_REDIS,
    ENABLE_GRACEFUL_DEGRADATION,
)

logger = logging.getLogger(__name__)

# Global Redis client
redis_client = None
redis_pool = None


def init_redis() -> bool:
    """
    Initialize Redis connection pool.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    global redis_client, redis_pool
    
    if not ENABLE_REDIS:
        logger.warning("Redis disabled via ENABLE_REDIS=false")
        return False
    
    try:
        # Create connection pool
        redis_pool = ConnectionPool(**REDIS_POOL_CONFIG)
        
        # Create Redis client
        redis_client = redis.Redis(connection_pool=redis_pool)
        
        # Test connection
        redis_client.ping()
        
        logger.info(f"Redis initialized: {REDIS_POOL_CONFIG['host']}:{REDIS_POOL_CONFIG['port']}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {e}")
        if ENABLE_GRACEFUL_DEGRADATION:
            logger.warning("Continuing without Redis (graceful degradation)")
            redis_client = None
            return False
        raise


def get_redis() -> Optional[redis.Redis]:
    """Get Redis client instance."""
    return redis_client


def _hash_key(key: str) -> str:
    """Hash a key to create shorter cache keys."""
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _make_cache_key(category: str, key: str) -> str:
    """Create a namespaced cache key."""
    prefix_map = {
        'cache': CACHE_KEY_CACHE,
        'session': CACHE_KEY_SESSION,
        'ratelimit': CACHE_KEY_RATE_LIMIT,
        'model': CACHE_KEY_MODEL,
    }
    prefix = prefix_map.get(category, CACHE_KEY_PREFIX)
    return f"{prefix}:{_hash_key(key)}"


# ============================================================================
# CACHING FUNCTIONS
# ============================================================================

def cache_get(key: str, category: str = 'cache') -> Optional[Any]:
    """
    Get value from cache.
    
    Args:
        key: Cache key
        category: Cache category (cache, session, model, etc.)
    
    Returns:
        Cached value or None if not found
    """
    if not redis_client:
        return None
    
    try:
        cache_key = _make_cache_key(category, key)
        data = redis_client.get(cache_key)
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        logger.error(f"Cache get error: {e}")
        return None


def cache_set(key: str, value: Any, ttl: int = 300, category: str = 'cache') -> bool:
    """
    Set value in cache.
    
    Args:
        key: Cache key
        value: Value to cache
        ttl: Time to live in seconds
        category: Cache category
    
    Returns:
        True if successful, False otherwise
    """
    if not redis_client:
        return False
    
    try:
        cache_key = _make_cache_key(category, key)
        data = json.dumps(value)
        redis_client.setex(cache_key, ttl, data)
        return True
    except Exception as e:
        logger.error(f"Cache set error: {e}")
        return False


def cache_delete(key: str, category: str = 'cache') -> bool:
    """
    Delete value from cache.
    
    Args:
        key: Cache key
        category: Cache category
    
    Returns:
        True if successful, False otherwise
    """
    if not redis_client:
        return False
    
    try:
        cache_key = _make_cache_key(category, key)
        redis_client.delete(cache_key)
        return True
    except Exception as e:
        logger.error(f"Cache delete error: {e}")
        return False


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

def session_set(session_id: str, key: str, value: Any, ttl: int = 86400) -> bool:
    """Set session data."""
    return cache_set(f"{session_id}:{key}", value, ttl, 'session')


def session_get(session_id: str, key: str) -> Optional[Any]:
    """Get session data."""
    return cache_get(f"{session_id}:{key}", 'session')


def session_list(session_id: str, pattern: str = "*") -> List[str]:
    """List session keys."""
    if not redis_client:
        return []
    
    try:
        prefix = _make_cache_key('session', f"{session_id}:{pattern}")
        keys = redis_client.keys(prefix)
        return [k.decode() if isinstance(k, bytes) else k for k in keys]
    except Exception as e:
        logger.error(f"Session list error: {e}")
        return []


def session_clear(session_id: str) -> bool:
    """Clear all session data for a session."""
    if not redis_client:
        return False
    
    try:
        prefix = _make_cache_key('session', f"{session_id}:*")
        keys = redis_client.keys(prefix)
        if keys:
            redis_client.delete(*keys)
        return True
    except Exception as e:
        logger.error(f"Session clear error: {e}")
        return False


# ============================================================================
# RATE LIMITING
# ============================================================================

def check_rate_limit(identifier: str, limit: int = None, window: int = None) -> Tuple[bool, int]:
    """
    Check rate limit for an identifier.
    
    Args:
        identifier: Unique identifier (user_id, ip_address, etc.)
        limit: Maximum requests allowed
        window: Time window in seconds
    
    Returns:
        Tuple of (is_allowed, remaining_requests)
    """
    if not RATE_LIMIT_ENABLED or not redis_client:
        return (True, 999999)
    
    limit = limit or RATE_LIMIT_REQUESTS
    window = window or RATE_LIMIT_WINDOW
    
    try:
        key = _make_cache_key('ratelimit', identifier)
        
        # Get current count
        current = redis_client.get(key)
        if current is None:
            # First request in window
            redis_client.setex(key, window, 1)
            return (True, limit - 1)
        
        current = int(current)
        if current >= limit:
            # Rate limit exceeded
            return (False, 0)
        
        # Increment and return remaining
        new_count = redis_client.incr(key)
        ttl = redis_client.ttl(key)
        if ttl == -1:  # Key exists but no TTL (shouldn't happen but be safe)
            redis_client.expire(key, window)
        
        return (True, limit - new_count)
        
    except Exception as e:
        logger.error(f"Rate limit check error: {e}")
        # On error, allow the request (fail open)
        return (True, 999999)


# ============================================================================
# MODEL CACHING
# ============================================================================

def cache_model_metadata(model_name: str, metadata: Dict[str, Any], ttl: int = 3600) -> bool:
    """Cache model metadata."""
    return cache_set(f"metadata:{model_name}", metadata, ttl, 'model')


def get_model_metadata(model_name: str) -> Optional[Dict[str, Any]]:
    """Get cached model metadata."""
    return cache_get(f"metadata:{model_name}", 'model')


def cache_model_data(model_name: str, data: bytes, ttl: int = 86400) -> bool:
    """
    Cache model data (large binary data).
    
    Note: This is for small model configs. For large model files, use external storage.
    """
    if not redis_client:
        return False
    
    try:
        key = _make_cache_key('model', f"data:{model_name}")
        redis_client.setex(key, ttl, data)
        return True
    except Exception as e:
        logger.error(f"Model data cache error: {e}")
        return False


# ============================================================================
# HEALTH CHECK
# ============================================================================

async def check_redis_health() -> Tuple[bool, str]:
    """
    Check if Redis is healthy.
    
    Returns:
        Tuple of (is_healthy, message)
    """
    if not ENABLE_REDIS or redis_client is None:
        return (False, "Redis not enabled or not initialized")
    
    try:
        start_time = time.time()
        redis_client.ping()
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        return (True, f"Redis connection healthy (latency: {latency:.2f}ms)")
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return (False, f"Redis connection failed: {str(e)}")


# ============================================================================
# STATISTICS
# ============================================================================

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    if not redis_client:
        return {
            "enabled": False,
            "total_keys": 0,
            "memory_used": 0,
        }
    
    try:
        info = redis_client.info('memory')
        return {
            "enabled": True,
            "total_keys": redis_client.dbsize(),
            "memory_used": info.get('used_memory_human', 'N/A'),
            "max_memory": info.get('maxmemory_human', 'N/A'),
        }
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        return {
            "enabled": True,
            "error": str(e),
        }


# Initialize Redis on module import
if __name__ != "__main__":
    init_redis()
