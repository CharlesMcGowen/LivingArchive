#!/usr/bin/env python3
"""
Query Builder Utilities for EgoLlama
=====================================

ORM query builders to replace raw SQL with type-safe queries.
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar
from sqlalchemy import select, and_, or_
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from db_models import (
    LLaMAModel,
    LLaMAInferenceRequest,
    LLaMAPerformanceMetric,
    LLaMACacheEntry
)

logger = logging.getLogger(__name__)

ModelType = TypeVar('ModelType')


def build_model_query(
    session: Session,
    filters: Dict[str, Any] = None
) -> Any:
    """
    Build a filtered query for LLaMAModel.
    
    Args:
        session: SQLAlchemy session
        filters: Dict of filter conditions
        
    Returns:
        SQLAlchemy query object
    """
    try:
        query = session.query(LLaMAModel)
        
        if filters:
            # Filter by is_active
            if filters.get('is_active') is not None:
                query = query.filter(LLaMAModel.is_active == filters['is_active'])
            
            # Filter by name contains
            if filters.get('name_contains'):
                query = query.filter(LLaMAModel.name.ilike(f"%{filters['name_contains']}%"))
            
            # Filter by name exact match
            if filters.get('name'):
                query = query.filter(LLaMAModel.name == filters['name'])
            
            # Filter by path
            if filters.get('path'):
                query = query.filter(LLaMAModel.model_path == filters['path'])
        
        return query
        
    except Exception as e:
        logger.error(f"Error building model query: {e}")
        return session.query(LLaMAModel).filter(False)


async def async_build_model_query(
    session: AsyncSession,
    filters: Dict[str, Any] = None
) -> Any:
    """
    Build an async filtered query for LLaMAModel.
    
    Args:
        session: Async SQLAlchemy session
        filters: Dict of filter conditions
        
    Returns:
        SQLAlchemy select statement
    """
    try:
        stmt = select(LLaMAModel)
        
        if filters:
            conditions = []
            
            # Filter by is_active
            if filters.get('is_active') is not None:
                conditions.append(LLaMAModel.is_active == filters['is_active'])
            
            # Filter by name contains
            if filters.get('name_contains'):
                conditions.append(LLaMAModel.name.ilike(f"%{filters['name_contains']}%"))
            
            # Filter by name exact match
            if filters.get('name'):
                conditions.append(LLaMAModel.name == filters['name'])
            
            # Filter by path
            if filters.get('path'):
                conditions.append(LLaMAModel.model_path == filters['path'])
            
            if conditions:
                stmt = stmt.where(and_(*conditions))
        
        return stmt
        
    except Exception as e:
        logger.error(f"Error building async model query: {e}")
        return select(LLaMAModel).where(False)


def build_inference_query(
    session: Session,
    filters: Dict[str, Any] = None
) -> Any:
    """
    Build a filtered query for inference requests.
    
    Args:
        session: SQLAlchemy session
        filters: Dict of filter conditions
        
    Returns:
        SQLAlchemy query object
    """
    try:
        query = session.query(LLaMAInferenceRequest)
        
        if filters:
            # Filter by model_id
            if filters.get('model_id'):
                query = query.filter(LLaMAInferenceRequest.model_id == filters['model_id'])
            
            # Filter by status code
            if filters.get('status_code'):
                query = query.filter(LLaMAInferenceRequest.status_code == filters['status_code'])
            
            # Filter by min latency
            if filters.get('min_latency'):
                query = query.filter(LLaMAInferenceRequest.latency >= filters['min_latency'])
            
            # Filter by date range
            if filters.get('start_date'):
                query = query.filter(LLaMAInferenceRequest.request_time >= filters['start_date'])
            
            if filters.get('end_date'):
                query = query.filter(LLaMAInferenceRequest.request_time <= filters['end_date'])
        
        # Default ordering by request_time desc
        query = query.order_by(LLaMAInferenceRequest.request_time.desc())
        
        return query
        
    except Exception as e:
        logger.error(f"Error building inference query: {e}")
        return session.query(LLaMAInferenceRequest).filter(False)


async def async_build_inference_query(
    session: AsyncSession,
    filters: Dict[str, Any] = None
) -> Any:
    """
    Build an async filtered query for inference requests.
    
    Args:
        session: Async SQLAlchemy session
        filters: Dict of filter conditions
        
    Returns:
        SQLAlchemy select statement
    """
    try:
        stmt = select(LLaMAInferenceRequest)
        
        if filters:
            conditions = []
            
            # Filter by model_id
            if filters.get('model_id'):
                conditions.append(LLaMAInferenceRequest.model_id == filters['model_id'])
            
            # Filter by status code
            if filters.get('status_code'):
                conditions.append(LLaMAInferenceRequest.status_code == filters['status_code'])
            
            # Filter by min latency
            if filters.get('min_latency'):
                conditions.append(LLaMAInferenceRequest.latency >= filters['min_latency'])
            
            # Filter by date range
            if filters.get('start_date'):
                conditions.append(LLaMAInferenceRequest.request_time >= filters['start_date'])
            
            if filters.get('end_date'):
                conditions.append(LLaMAInferenceRequest.request_time <= filters['end_date'])
            
            if conditions:
                stmt = stmt.where(and_(*conditions))
        
        # Default ordering by request_time desc
        stmt = stmt.order_by(LLaMAInferenceRequest.request_time.desc())
        
        return stmt
        
    except Exception as e:
        logger.error(f"Error building async inference query: {e}")
        return select(LLaMAInferenceRequest).where(False)


def build_cache_query(
    session: Session,
    filters: Dict[str, Any] = None
) -> Any:
    """
    Build a filtered query for cache entries.
    
    Args:
        session: SQLAlchemy session
        filters: Dict of filter conditions
        
    Returns:
        SQLAlchemy query object
    """
    try:
        query = session.query(LLaMACacheEntry)
        
        if filters:
            # Filter by model_id
            if filters.get('model_id'):
                query = query.filter(LLaMACacheEntry.model_id == filters['model_id'])
            
            # Filter by min hit count
            if filters.get('min_hit_count'):
                query = query.filter(LLaMACacheEntry.hit_count >= filters['min_hit_count'])
            
            # Filter by prompt contains
            if filters.get('prompt_contains'):
                query = query.filter(LLaMACacheEntry.prompt.ilike(f"%{filters['prompt_contains']}%"))
        
        # Default ordering by last_accessed desc
        query = query.order_by(LLaMACacheEntry.last_accessed.desc())
        
        return query
        
    except Exception as e:
        logger.error(f"Error building cache query: {e}")
        return session.query(LLaMACacheEntry).filter(False)


async def async_build_cache_query(
    session: AsyncSession,
    filters: Dict[str, Any] = None
) -> Any:
    """
    Build an async filtered query for cache entries.
    
    Args:
        session: Async SQLAlchemy session
        filters: Dict of filter conditions
        
    Returns:
        SQLAlchemy select statement
    """
    try:
        stmt = select(LLaMACacheEntry)
        
        if filters:
            conditions = []
            
            # Filter by model_id
            if filters.get('model_id'):
                conditions.append(LLaMACacheEntry.model_id == filters['model_id'])
            
            # Filter by min hit count
            if filters.get('min_hit_count'):
                conditions.append(LLaMACacheEntry.hit_count >= filters['min_hit_count'])
            
            # Filter by prompt contains
            if filters.get('prompt_contains'):
                conditions.append(LLaMACacheEntry.prompt.ilike(f"%{filters['prompt_contains']}%"))
            
            if conditions:
                stmt = stmt.where(and_(*conditions))
        
        # Default ordering by last_accessed desc
        stmt = stmt.order_by(LLaMACacheEntry.last_accessed.desc())
        
        return stmt
        
    except Exception as e:
        logger.error(f"Error building async cache query: {e}")
        return select(LLaMACacheEntry).where(False)


