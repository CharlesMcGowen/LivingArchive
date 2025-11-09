#!/usr/bin/env python3
"""
EgoLlama Configuration
=====================

Centralized configuration for database, Redis, caching, and rate limiting.
"""

import os
from typing import Optional

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# Base DATABASE_URL - add sslmode=disable to prevent SSL negotiation issues
_base_url = os.environ.get(
    'EGOLLAMA_DATABASE_URL',
    'postgresql://postgres:postgres@localhost:5432/ego'
)
# Ensure sslmode=disable is in URL if not already present
if 'sslmode' not in _base_url:
    separator = '&' if '?' in _base_url else '?'
    DATABASE_URL = f"{_base_url}{separator}sslmode=disable"
else:
    DATABASE_URL = _base_url

# Connection pool settings
DB_POOL_SIZE = int(os.environ.get('DB_POOL_SIZE', '10'))
DB_MAX_OVERFLOW = int(os.environ.get('DB_MAX_OVERFLOW', '20'))
DB_POOL_TIMEOUT = int(os.environ.get('DB_POOL_TIMEOUT', '30'))
DB_POOL_RECYCLE = int(os.environ.get('DB_POOL_RECYCLE', '3600'))

# Connection options
DB_ECHO = os.environ.get('DB_ECHO', 'False').lower() == 'true'
DB_AUTOCOMMIT = os.environ.get('DB_AUTOCOMMIT', 'True').lower() == 'true'

# ============================================================================
# REDIS CONFIGURATION
# ============================================================================

REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
REDIS_DB = int(os.environ.get('REDIS_DB', '0'))
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')
REDIS_SOCKET_TIMEOUT = int(os.environ.get('REDIS_SOCKET_TIMEOUT', '5'))
REDIS_MAX_CONNECTIONS = int(os.environ.get('REDIS_MAX_CONNECTIONS', '50'))

# Redis connection pool
REDIS_POOL_CONFIG = {
    'host': REDIS_HOST,
    'port': REDIS_PORT,
    'db': REDIS_DB,
    'password': REDIS_PASSWORD,
    'socket_timeout': REDIS_SOCKET_TIMEOUT,
    'max_connections': REDIS_MAX_CONNECTIONS,
    'decode_responses': False,
}

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

# Cache TTLs (in seconds)
CACHE_TTL_SHORT = int(os.environ.get('CACHE_TTL_SHORT', '5'))       # Health checks
CACHE_TTL_MEDIUM = int(os.environ.get('CACHE_TTL_MEDIUM', '300'))   # API responses
CACHE_TTL_LONG = int(os.environ.get('CACHE_TTL_LONG', '3600'))      # Model metadata
CACHE_TTL_SESSION = int(os.environ.get('CACHE_TTL_SESSION', '86400'))  # Sessions (24h)

# Cache key prefixes
CACHE_KEY_PREFIX = 'egollama'
CACHE_KEY_CACHE = f'{CACHE_KEY_PREFIX}:cache'
CACHE_KEY_SESSION = f'{CACHE_KEY_PREFIX}:session'
CACHE_KEY_RATE_LIMIT = f'{CACHE_KEY_PREFIX}:ratelimit'
CACHE_KEY_MODEL = f'{CACHE_KEY_PREFIX}:model'

# ============================================================================
# RATE LIMITING CONFIGURATION
# ============================================================================

RATE_LIMIT_ENABLED = os.environ.get('RATE_LIMIT_ENABLED', 'True').lower() == 'true'
RATE_LIMIT_REQUESTS = int(os.environ.get('RATE_LIMIT_REQUESTS', '100'))
RATE_LIMIT_WINDOW = int(os.environ.get('RATE_LIMIT_WINDOW', '3600'))  # 1 hour
RATE_LIMIT_BLOCK_DURATION = int(os.environ.get('RATE_LIMIT_BLOCK_DURATION', '3600'))

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
LOG_FORMAT = os.environ.get('LOG_FORMAT', 'json')  # json or text
LOG_FILE = os.environ.get('LOG_FILE', 'logs/egollama.log')

# ============================================================================
# GATEWAY CONFIGURATION
# ============================================================================

GATEWAY_HOST = os.environ.get('GATEWAY_HOST', '0.0.0.0')
GATEWAY_PORT = int(os.environ.get('GATEWAY_PORT', '8082'))
GATEWAY_WORKERS = int(os.environ.get('GATEWAY_WORKERS', '4'))
GATEWAY_RELOAD = os.environ.get('GATEWAY_RELOAD', 'False').lower() == 'true'

# Request limits
MAX_REQUEST_SIZE = int(os.environ.get('MAX_REQUEST_SIZE', '10485760'))  # 10MB
MAX_RESPONSE_SIZE = int(os.environ.get('MAX_RESPONSE_SIZE', '104857600'))  # 100MB

# ============================================================================
# HEALTH CHECK CONFIGURATION
# ============================================================================

HEALTH_CHECK_INTERVAL = int(os.environ.get('HEALTH_CHECK_INTERVAL', '60'))
HEALTH_CHECK_TIMEOUT = int(os.environ.get('HEALTH_CHECK_TIMEOUT', '5'))
ENABLE_DB_HEALTH_CHECK = os.environ.get('ENABLE_DB_HEALTH_CHECK', 'True').lower() == 'true'
ENABLE_REDIS_HEALTH_CHECK = os.environ.get('ENABLE_REDIS_HEALTH_CHECK', 'True').lower() == 'true'

# ============================================================================
# FEATURE FLAGS
# ============================================================================

ENABLE_POSTGRESQL = os.environ.get('ENABLE_POSTGRESQL', 'True').lower() == 'true'
ENABLE_REDIS = os.environ.get('ENABLE_REDIS', 'True').lower() == 'true'
ENABLE_GRACEFUL_DEGRADATION = os.environ.get('ENABLE_GRACEFUL_DEGRADATION', 'True').lower() == 'true'

# ============================================================================
# CIRCUIT BREAKER CONFIGURATION
# ============================================================================

CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.environ.get('CIRCUIT_BREAKER_FAILURE_THRESHOLD', '5'))
CIRCUIT_BREAKER_TIMEOUT = int(os.environ.get('CIRCUIT_BREAKER_TIMEOUT', '60'))
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = int(os.environ.get('CIRCUIT_BREAKER_RECOVERY_TIMEOUT', '300'))
