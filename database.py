#!/usr/bin/env python3
"""
Database Module
===============

SQLAlchemy database configuration with async support for EgoLlama.
"""

import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from sqlalchemy.pool import NullPool
from sqlalchemy import text

from config import (
    DATABASE_URL,
    DB_POOL_SIZE,
    DB_MAX_OVERFLOW,
    DB_POOL_TIMEOUT,
    DB_POOL_RECYCLE,
    DB_ECHO,
    ENABLE_POSTGRESQL,
    ENABLE_GRACEFUL_DEGRADATION,
)

logger = logging.getLogger(__name__)

# Base model class
Base = declarative_base()

# Global engine and session factory
engine = None
async_engine = None
AsyncSessionLocal = None
SessionLocal = None


def init_database() -> bool:
    """
    Initialize database connections.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    global engine, async_engine, AsyncSessionLocal, SessionLocal
    
    # Skip if already initialized
    if engine is not None and async_engine is not None:
        logger.debug("Database already initialized, skipping")
        return True
    
    if not ENABLE_POSTGRESQL:
        logger.warning("PostgreSQL disabled via ENABLE_POSTGRESQL=false")
        return False
    
    try:
        # Convert postgresql:// to postgresql+asyncpg:// for async
        # Remove query params and pass SSL via connect_args for asyncpg
        base_url = DATABASE_URL.split('?')[0]
        async_database_url = base_url.replace('postgresql://', 'postgresql+asyncpg://')
        
        # Extract SSL mode for asyncpg (uses ssl=False, not sslmode)
        connect_args = {}
        if 'sslmode=disable' in DATABASE_URL:
            connect_args['ssl'] = False
        # Add timeout for asyncpg stability
        connect_args['command_timeout'] = 60
        
        # Create async engine
        async_engine = create_async_engine(
            async_database_url,
            pool_size=DB_POOL_SIZE,
            max_overflow=DB_MAX_OVERFLOW,
            pool_timeout=DB_POOL_TIMEOUT,
            pool_recycle=DB_POOL_RECYCLE,
            echo=DB_ECHO,
            pool_pre_ping=True,  # Verify connections before using
            future=True,
            connect_args=connect_args if connect_args else {},
        )
        
        # Create async session factory
        AsyncSessionLocal = async_sessionmaker(
            async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
        
        # Create sync engine for compatibility
        # Add connect_args to handle scram-sha-256 and connection issues
        connect_args = {
            'connect_timeout': 10,
            'application_name': 'EgoLlama',
        }
        
        # Add sslmode=disable if not explicitly set in URL
        if 'sslmode' not in DATABASE_URL:
            connect_args['sslmode'] = 'disable'
        
        engine = create_engine(
            DATABASE_URL,
            pool_size=DB_POOL_SIZE,
            max_overflow=DB_MAX_OVERFLOW,
            pool_timeout=DB_POOL_TIMEOUT,
            pool_recycle=DB_POOL_RECYCLE,
            echo=DB_ECHO,
            pool_pre_ping=True,
            future=True,
            connect_args=connect_args,
        )
        
        # Create sync session factory
        SessionLocal = sessionmaker(
            bind=engine,
            autocommit=False,
            autoflush=False,
        )
        
        logger.info(f"Database initialized: {DATABASE_URL}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        if ENABLE_GRACEFUL_DEGRADATION:
            logger.warning("Continuing without database (graceful degradation)")
            return False
        raise


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function for FastAPI to get async database session.
    
    Yields:
        AsyncSession: Database session
    """
    if AsyncSessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Database commit error, rolling back: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_db():
    """
    Dependency function to get sync database session.
    
    Yields:
        Session: Database session
    """
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def check_database_health() -> tuple[bool, str]:
    """
    Check if database is healthy.
    
    Returns:
        tuple: (is_healthy, message)
    """
    if not ENABLE_POSTGRESQL or async_engine is None:
        return (False, "Database not enabled or not initialized")
    
    try:
        async with async_engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            result.fetchone()
        
        return (True, "Database connection healthy")
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return (False, f"Database connection failed: {str(e)}")


async def close_database():
    """
    Close all database connections.
    """
    global engine, async_engine
    
    if async_engine:
        await async_engine.dispose()
        logger.info("Async database engine closed")
    
    if engine:
        engine.dispose()
        logger.info("Sync database engine closed")


# Initialize database on module import
if __name__ != "__main__":
    init_database()
