#!/usr/bin/env python3
"""
Pydantic Schemas for EgoLlama
==============================

Response models with ORM mode support for automatic SQLAlchemy serialization.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any


class ModelResponse(BaseModel):
    """Model response schema with ORM mode"""
    id: int
    name: str
    model_path: str
    loaded_at: Optional[datetime] = None
    is_active: bool = False
    config: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True  # ORM mode for SQLAlchemy


class InferenceRequestResponse(BaseModel):
    """Inference request response schema"""
    id: int
    model_id: int
    request_time: datetime
    response_time: Optional[datetime] = None
    prompt: str
    response: Optional[str] = None
    request_tokens: int = 0
    response_tokens: int = 0
    latency: Optional[float] = None
    status_code: int = 200
    error_message: Optional[str] = None
    model: Optional[ModelResponse] = None  # Nested relationship

    class Config:
        from_attributes = True


class PerformanceMetricResponse(BaseModel):
    """Performance metric response schema"""
    id: int
    model_id: int
    metric_type: str
    metric_value: float
    recorded_at: datetime
    context: Optional[Dict[str, Any]] = None
    model: Optional[ModelResponse] = None

    class Config:
        from_attributes = True


class CacheEntryResponse(BaseModel):
    """Cache entry response schema"""
    id: int
    model_id: int
    prompt: str
    response: str
    created_at: datetime
    last_accessed: datetime
    hit_count: int = 0
    extra_metadata: Optional[Dict[str, Any]] = None
    model: Optional[ModelResponse] = None

    class Config:
        from_attributes = True


class PromptTemplateResponse(BaseModel):
    """Prompt template response schema"""
    id: int
    template_name: str
    template_content: str
    recommended_model_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    recommended_model: Optional[ModelResponse] = None

    class Config:
        from_attributes = True


class PaginatedResponse(BaseModel):
    """Generic paginated response wrapper"""
    items: List[Any] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 50
    has_next: bool = False
    has_previous: bool = False

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool = False
    gpu_available: bool = False
    quantization: Optional[str] = None
    total_inferences: int = 0
    average_tokens_per_second: float = 0.0


class PerformanceStatsResponse(BaseModel):
    """Performance statistics response"""
    timestamp: str
    cpu_usage: float
    memory_usage: float
    gpu_usage: float = 0.0
    active_models: int = 0
    total_requests: int = 0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    model_loaded: bool = False
    gpu_available: bool = False


# Request models for filtering/pagination
class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(1, ge=1, description="Page number (1-based)")
    page_size: int = Field(50, ge=1, le=100, description="Items per page")


class ModelFilterParams(BaseModel):
    """Model filter parameters"""
    is_active: Optional[bool] = None
    name_contains: Optional[str] = None
    page: int = Field(1, ge=1)
    page_size: int = Field(50, ge=1, le=100)


class KnowledgeSearchParams(BaseModel):
    """Knowledge search parameters"""
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(5, ge=1, le=100)
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


