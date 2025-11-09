#!/usr/bin/env python3
"""
Pagination Utilities for EgoLlama
==================================

Utilities for paginating SQLAlchemy queries (sync and async).
"""

import logging
from typing import Any, Dict, List, Optional
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from schemas import PaginatedResponse

logger = logging.getLogger(__name__)


def paginate_query(
    session: Session,
    query: Any,
    page: int = 1,
    page_size: int = 50
) -> Dict[str, Any]:
    """
    Paginate a synchronous SQLAlchemy query.
    
    Args:
        session: SQLAlchemy session
        query: SQLAlchemy query object
        page: Page number (1-based)
        page_size: Items per page
        
    Returns:
        Dict with items, total, pagination metadata
    """
    try:
        # Get total count
        total = query.count()
        
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Get items
        items = query.offset(offset).limit(page_size).all()
        
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
        logger.error(f"Error in paginate_query: {e}")
        return {
            "items": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
            "has_next": False,
            "has_previous": False
        }


async def async_paginate_query(
    session: AsyncSession,
    stmt: Any,
    page: int = 1,
    page_size: int = 50
) -> Dict[str, Any]:
    """
    Paginate an async SQLAlchemy select statement.
    
    Args:
        session: Async SQLAlchemy session
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
        logger.error(f"Error in async_paginate_query: {e}")
        return {
            "items": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
            "has_next": False,
            "has_previous": False
        }


def create_paginated_response(
    items: List[Any],
    total: int,
    page: int,
    page_size: int
) -> PaginatedResponse:
    """
    Create a PaginatedResponse from results.
    
    Args:
        items: List of items for current page
        total: Total number of items
        page: Current page number
        page_size: Items per page
        
    Returns:
        PaginatedResponse object
    """
    has_next = (page * page_size) < total
    has_previous = page > 1
    
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        has_next=has_next,
        has_previous=has_previous
    )


