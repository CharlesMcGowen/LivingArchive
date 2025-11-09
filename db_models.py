#!/usr/bin/env python3
"""
SQLAlchemy Models for EgoLlama
==============================

Database models for tracking models, requests, metrics, cache, and templates.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, Float, BigInteger,
    ForeignKey, JSON, Index, Enum as SQLEnum
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
import uuid
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database import Base


class LLaMAModel(Base):
    """
    Represents a loaded LLaMA model in the gateway.
    Tracks model name, path, and loading status.
    """
    __tablename__ = 'llama_models'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    model_path = Column(String(255), nullable=False)
    loaded_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=False, nullable=False, index=True)
    config = Column(JSON, default=dict, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    inference_requests = relationship("LLaMAInferenceRequest", back_populates="model", cascade="all, delete-orphan")
    performance_metrics = relationship("LLaMAPerformanceMetric", back_populates="model", cascade="all, delete-orphan")
    cache_entries = relationship("LLaMACacheEntry", back_populates="model", cascade="all, delete-orphan")
    prompt_templates = relationship("LLaMAPromptTemplate", back_populates="recommended_model")
    
    def __str__(self) -> str:
        return self.name
    
    def mark_loaded(self):
        """Marks the model as loaded and updates the timestamp."""
        self.loaded_at = datetime.utcnow()
        self.is_active = True
    
    def mark_unloaded(self):
        """Marks the model as unloaded."""
        self.is_active = False


class LLaMAInferenceRequest(Base):
    """
    Tracks individual inference requests made to the LLaMA models.
    Includes request details, timestamps, and model association.
    """
    __tablename__ = 'llama_inference_requests'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('llama_models.id'), nullable=False, index=True)
    request_time = Column(DateTime, default=func.now(), nullable=False, index=True)
    response_time = Column(DateTime, nullable=True)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=True)
    request_tokens = Column(Integer, default=0, nullable=False)
    response_tokens = Column(Integer, default=0, nullable=False)
    latency = Column(Float, nullable=True)
    status_code = Column(Integer, default=200, nullable=False)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    model = relationship("LLaMAModel", back_populates="inference_requests")
    
    # Composite index for efficient queries
    __table_args__ = (
        Index('llama_inference_model_time_idx', 'model_id', 'request_time'),
    )
    
    def __str__(self) -> str:
        return f"Request to {self.model.name} at {self.request_time}"
    
    def calculate_latency(self) -> None:
        """Calculates the latency of the request."""
        if self.response_time and self.request_time:
            self.latency = (self.response_time - self.request_time).total_seconds()
    
    def mark_completed(self, response: str, response_tokens: int, status_code: int = 200) -> None:
        """Marks the request as completed."""
        self.response_time = datetime.utcnow()
        self.response = response
        self.response_tokens = response_tokens
        self.status_code = status_code
        self.calculate_latency()


class LLaMAPerformanceMetric(Base):
    """
    Tracks GPU performance metrics during LLaMA model inference.
    Provides data for optimization and resource management.
    """
    __tablename__ = 'llama_performance_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('llama_models.id'), nullable=False, index=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False, index=True)
    gpu_utilization = Column(Float, default=0.0, nullable=False)
    memory_used = Column(BigInteger, default=0, nullable=False)  # bytes
    memory_total = Column(BigInteger, default=0, nullable=False)  # bytes
    power_usage = Column(Float, default=0.0, nullable=False)  # watts
    gpu_temperature = Column(Float, default=0.0, nullable=False)  # celsius
    
    # Relationships
    model = relationship("LLaMAModel", back_populates="performance_metrics")
    
    # Composite index for efficient queries
    __table_args__ = (
        Index('llama_perf_model_time_idx', 'model_id', 'timestamp'),
    )
    
    def __str__(self) -> str:
        return f"Metrics for {self.model.name} at {self.timestamp}"


class LLaMACacheEntry(Base):
    """
    Stores cached responses for LLaMA model inference requests.
    Improves performance by serving previously generated responses.
    """
    __tablename__ = 'llama_cache_entries'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('llama_models.id'), nullable=False, index=True)
    prompt = Column(Text, nullable=False, index=True)
    response = Column(Text, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    last_accessed = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False, index=True)
    hit_count = Column(Integer, default=0, nullable=False)
    extra_metadata = Column(JSON, default=dict, nullable=True)
    
    # Relationships
    model = relationship("LLaMAModel", back_populates="cache_entries")
    
    # Composite index for faster lookups
    __table_args__ = (
        Index('llama_cache_model_prompt_idx', 'model_id', 'prompt'),
    )
    
    def __str__(self) -> str:
        return f"Cache entry for {self.model.name} - {self.prompt[:50]}..."
    
    def update_access(self) -> None:
        """Updates the last accessed timestamp and increments the hit count."""
        self.last_accessed = datetime.utcnow()
        self.hit_count += 1


class LLaMAGatewayHealth(Base):
    """
    Track LLaMA Gateway health over time.
    
    Hybrid model: core health metrics + flexible JSON for detailed stats.
    """
    __tablename__ = 'llama_gateway_health'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False, index=True)
    status = Column(String(20), nullable=False, index=True)  # healthy, degraded, unhealthy
    gpu_available = Column(Boolean, nullable=False)
    cache_hit_rate = Column(Float, nullable=False)
    active_models = Column(Integer, default=0, nullable=False)
    metrics = Column(JSON, default=dict, nullable=True)
    
    # Composite index for efficient queries
    __table_args__ = (
        Index('llama_health_status_time_idx', 'status', 'timestamp'),
    )
    
    @property
    def is_healthy(self) -> bool:
        """Check if gateway is healthy"""
        return self.status == 'healthy'
    
    def __str__(self) -> str:
        return f"{self.status} at {self.timestamp}"


class LLaMAPromptTemplate(Base):
    """
    Reusable prompt templates for common tasks.
    
    Hybrid model: structured template info + flexible configuration.
    """
    __tablename__ = 'llama_prompt_templates'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), unique=True, nullable=False)
    category = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=False)
    template_text = Column(Text, nullable=False)
    recommended_model_id = Column(Integer, ForeignKey('llama_models.id'), nullable=True, index=True)
    recommended_temperature = Column(Float, default=0.3, nullable=False)
    recommended_max_tokens = Column(Integer, default=512, nullable=False)
    config = Column(JSON, default=dict, nullable=True)
    use_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    recommended_model = relationship("LLaMAModel", back_populates="prompt_templates")
    
    # Index for ordering by use count
    __table_args__ = (
        Index('llama_template_use_count_idx', 'use_count'),
    )
    
    def __str__(self) -> str:
        return f"{self.name} ({self.category})"
    
    def increment_usage(self) -> None:
        """Increment usage counter"""
        self.use_count += 1

