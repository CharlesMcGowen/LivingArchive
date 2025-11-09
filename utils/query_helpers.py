#!/usr/bin/env python3
"""
Async Query Helpers for EgoLlama
================================

Async query utilities for database operations with SQLAlchemy.
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from sqlalchemy.orm import joinedload, selectinload
from fastapi import HTTPException, status

from db_models import (
    LLaMAModel,
    LLaMAInferenceRequest,
    LLaMAPerformanceMetric,
    LLaMACacheEntry,
    LLaMAPromptTemplate
)

logger = logging.getLogger(__name__)

# Generic type for models
ModelType = TypeVar('ModelType')


async def async_get_or_404(
    session: AsyncSession,
    model: Type[ModelType],
    item_id: int,
    relationship_options: Optional[List] = None
) -> ModelType:
    """
    Fetch a single item by ID or raise 404.
    
    Args:
        session: Async database session
        model: SQLAlchemy model class
        item_id: ID to fetch
        relationship_options: Optional list of relationship load options
        
    Returns:
        Model instance
        
    Raises:
        HTTPException: 404 if not found
    """
    try:
        stmt = select(model).filter(model.id == item_id)
        
        # Add relationship loading if specified
        if relationship_options:
            stmt = stmt.options(*relationship_options)
        
        result = await session.execute(stmt)
        item = result.scalar_one_or_none()
        
        if item is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{model.__name__} with id {item_id} not found"
            )
        
        return item
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching {model.__name__} {item_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching {model.__name__}"
        )


async def async_paginate_query(
    session: AsyncSession,
    stmt: Any,
    page: int = 1,
    page_size: int = 50
) -> Dict[str, Any]:
    """
    Paginate a query with total count and metadata.
    
    Args:
        session: Async database session
        stmt: SQLAlchemy select statement
        page: Page number (1-based)
        page_size: Items per page
        
    Returns:
        Dict with items, total, pagination metadata
    """
    try:
        # Get total count
        count_stmt = select(func.count()).select_from(stmt.subquery())
        count_result = await session.execute(count_stmt)
        total = count_result.scalar() or 0
        
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Apply pagination
        paginated_stmt = stmt.offset(offset).limit(page_size)
        
        # Execute query
        result = await session.execute(paginated_stmt)
        items = result.scalars().all()
        
        # Calculate pagination metadata
        has_next = (page * page_size) < total
        has_previous = page > 1
        
        return {
            "items": list(items),
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_next": has_next,
            "has_previous": has_previous
        }
        
    except Exception as e:
        logger.error(f"Error paginating query: {e}")
        return {
            "items": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
            "has_next": False,
            "has_previous": False
        }


async def async_filter_models(
    session: AsyncSession,
    model: Type[ModelType],
    filters: Dict[str, Any]
) -> Any:
    """
    Build and execute filtered query dynamically.
    
    Args:
        session: Async database session
        model: SQLAlchemy model class
        filters: Dict of filter conditions
        
    Returns:
        SQLAlchemy select statement
    """
    try:
        stmt = select(model)
        
        # Apply filters dynamically
        for field, value in filters.items():
            if hasattr(model, field) and value is not None:
                stmt = stmt.filter(getattr(model, field) == value)
        
        return stmt
        
    except Exception as e:
        logger.error(f"Error building filter query: {e}")
        return select(model).filter(False)  # Return empty result


async def async_bulk_update(
    session: AsyncSession,
    model: Type[ModelType],
    updates: List[Dict[str, Any]],
    batch_size: int = 1000
) -> int:
    """
    Perform bulk updates in batches.
    
    Args:
        session: Async database session
        model: SQLAlchemy model class
        updates: List of update dicts with 'id' and other fields
        batch_size: Number of updates per batch
        
    Returns:
        Number of rows updated
    """
    try:
        total_updated = 0
        
        # Process in batches
        for i in range(0, len(updates), batch_size):
            batch = updates[i:i + batch_size]
            
            for update in batch:
                item_id = update.pop('id')
                await session.execute(
                    select(model).where(model.id == item_id).values(**update)
                )
            
            total_updated += len(batch)
            
            # Commit batch
            await session.commit()
        
        return total_updated
        
    except Exception as e:
        logger.error(f"Error in bulk update: {e}")
        await session.rollback()
        return 0


async def get_inference_with_model(
    session: AsyncSession,
    inference_id: int
) -> Optional[LLaMAInferenceRequest]:
    """
    Get inference request with related model loaded.
    
    Args:
        session: Async database session
        inference_id: Inference request ID
        
    Returns:
        InferenceRequest with model loaded
    """
    stmt = (
        select(LLaMAInferenceRequest)
        .options(joinedload(LLaMAInferenceRequest.model))
        .filter(LLaMAInferenceRequest.id == inference_id)
    )
    
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_model_inferences(
    session: AsyncSession,
    model_id: int,
    limit: int = 10
) -> List[LLaMAInferenceRequest]:
    """
    Get recent inference requests for a model.
    
    Args:
        session: Async database session
        model_id: Model ID
        limit: Max results
        
    Returns:
        List of inference requests
    """
    stmt = (
        select(LLaMAInferenceRequest)
        .filter(LLaMAInferenceRequest.model_id == model_id)
        .order_by(LLaMAInferenceRequest.request_time.desc())
        .limit(limit)
    )
    
    result = await session.execute(stmt)
    return result.scalars().all()


async def get_cache_entries_with_model(
    session: AsyncSession,
    model_id: Optional[int] = None,
    limit: int = 100
) -> List[LLaMACacheEntry]:
    """
    Get cache entries with optional model filter.
    
    Args:
        session: Async database session
        model_id: Optional model ID filter
        limit: Max results
        
    Returns:
        List of cache entries
    """
    stmt = (
        select(LLaMACacheEntry)
        .options(selectinload(LLaMACacheEntry.model))
    )
    
    if model_id:
        stmt = stmt.filter(LLaMACacheEntry.model_id == model_id)
    
    stmt = stmt.order_by(LLaMACacheEntry.last_accessed.desc()).limit(limit)
    
    result = await session.execute(stmt)
    return result.scalars().all()


