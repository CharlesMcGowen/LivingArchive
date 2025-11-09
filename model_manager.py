#!/usr/bin/env python3
"""
Model Management Service for LLaMA Gateway
==========================================

Manages AI models for the GPU-driven LLaMA gateway.
Handles model loading, unloading, and metadata.

Author: EGO Revolution Team
Version: 1.0.0 - Phase 1 Enhancement
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# Add GPU_Accelerator to path
sys.path.insert(0, str(Path(__file__).parent.parent / "GPU_Accelerator"))

logger = logging.getLogger(__name__)

class ModelInfo:
    """Model information container"""
    
    def __init__(self, name: str, size_gb: float = 0.0, loaded: bool = False, 
                 memory_usage: float = 0.0, last_accessed: float = 0.0):
        self.name = name
        self.size_gb = size_gb
        self.loaded = loaded
        self.memory_usage = memory_usage
        self.last_accessed = last_accessed
        self.created_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "size_gb": self.size_gb,
            "loaded": self.loaded,
            "memory_usage": self.memory_usage,
            "last_accessed": self.last_accessed,
            "created_at": self.created_at
        }

class ModelManager:
    """Model management service"""
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self.loaded_models: Dict[str, ModelInfo] = {}
        self.max_memory_gb = 8.0  # AMD Radeon RX 5700 XT has 8GB VRAM
        self.current_memory_usage = 0.0
        
        # Initialize with default models
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize with default models"""
        default_models = [
            ("llama3.2:1b", 1.0),
            ("llama3.2:3b", 3.0),
            ("llama3.2:8b", 8.0),
            ("mistral:7b", 7.0),
            ("phi3:3.8b", 3.8),
            ("codellama:7b", 7.0),
            ("deepseek-coder:6.7b", 6.7),
            ("wizardcoder:7b", 7.0),
            ("qwen2.5:7b", 7.0),
            ("gemma2:9b", 9.0)
        ]
        
        for name, size in default_models:
            self.models[name] = ModelInfo(name, size)
        
        logger.info(f"Initialized {len(default_models)} default models")
    
    async def list_models(self) -> Dict[str, Any]:
        """List all available models"""
        try:
            model_list = []
            for model_info in self.models.values():
                model_dict = model_info.to_dict()
                model_list.append(model_dict)
            
            return {
                "models": model_list,
                "total_models": len(self.models),
                "loaded_models": len(self.loaded_models),
                "available_memory_gb": self.max_memory_gb - self.current_memory_usage,
                "total_memory_gb": self.max_memory_gb
            }
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return {"error": str(e)}
    
    async def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a model into memory"""
        try:
            if model_name not in self.models:
                return {"error": f"Model {model_name} not found"}
            
            model_info = self.models[model_name]
            
            if model_info.loaded:
                return {"message": f"Model {model_name} is already loaded", "status": "already_loaded"}
            
            # Check memory availability
            if self.current_memory_usage + model_info.size_gb > self.max_memory_gb:
                # Try to unload other models
                await self._free_memory(model_info.size_gb)
                
                if self.current_memory_usage + model_info.size_gb > self.max_memory_gb:
                    return {"error": f"Insufficient memory. Need {model_info.size_gb}GB, have {self.max_memory_gb - self.current_memory_usage:.1f}GB"}
            
            # Simulate model loading
            await asyncio.sleep(0.1)  # Simulate loading time
            
            # Update model status
            model_info.loaded = True
            model_info.last_accessed = time.time()
            model_info.memory_usage = model_info.size_gb
            self.loaded_models[model_name] = model_info
            self.current_memory_usage += model_info.size_gb
            
            logger.info(f"Model {model_name} loaded successfully")
            return {
                "message": f"Model {model_name} loaded successfully",
                "model": model_info.to_dict(),
                "memory_usage": self.current_memory_usage,
                "status": "loaded"
            }
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return {"error": str(e)}
    
    async def unload_model(self, model_name: str) -> Dict[str, Any]:
        """Unload a model from memory"""
        try:
            if model_name not in self.loaded_models:
                return {"error": f"Model {model_name} is not loaded"}
            
            model_info = self.loaded_models[model_name]
            
            # Simulate model unloading
            await asyncio.sleep(0.05)  # Simulate unloading time
            
            # Update model status
            model_info.loaded = False
            model_info.memory_usage = 0.0
            self.current_memory_usage -= model_info.size_gb
            del self.loaded_models[model_name]
            
            logger.info(f"Model {model_name} unloaded successfully")
            return {
                "message": f"Model {model_name} unloaded successfully",
                "memory_usage": self.current_memory_usage,
                "status": "unloaded"
            }
            
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            return {"error": str(e)}
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information"""
        try:
            if model_name not in self.models:
                return {"error": f"Model {model_name} not found"}
            
            model_info = self.models[model_name]
            return {
                "model": model_info.to_dict(),
                "status": "loaded" if model_info.loaded else "available"
            }
            
        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return {"error": str(e)}
    
    async def delete_model(self, model_name: str) -> Dict[str, Any]:
        """Delete a model"""
        try:
            if model_name not in self.models:
                return {"error": f"Model {model_name} not found"}
            
            # Unload if loaded
            if model_name in self.loaded_models:
                await self.unload_model(model_name)
            
            # Remove from models
            del self.models[model_name]
            
            logger.info(f"Model {model_name} deleted successfully")
            return {
                "message": f"Model {model_name} deleted successfully",
                "status": "deleted"
            }
            
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {e}")
            return {"error": str(e)}
    
    async def _free_memory(self, required_gb: float) -> bool:
        """Free memory by unloading models"""
        try:
            # Sort loaded models by last accessed (oldest first)
            sorted_models = sorted(
                self.loaded_models.items(),
                key=lambda x: x[1].last_accessed
            )
            
            freed_memory = 0.0
            for model_name, model_info in sorted_models:
                if freed_memory >= required_gb:
                    break
                
                await self.unload_model(model_name)
                freed_memory += model_info.size_gb
            
            return freed_memory >= required_gb
            
        except Exception as e:
            logger.error(f"Error freeing memory: {e}")
            return False
    
    async def get_memory_status(self) -> Dict[str, Any]:
        """Get memory status"""
        return {
            "current_usage_gb": self.current_memory_usage,
            "max_memory_gb": self.max_memory_gb,
            "available_memory_gb": self.max_memory_gb - self.current_memory_usage,
            "loaded_models": len(self.loaded_models),
            "total_models": len(self.models)
        }
    
    async def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage"""
        try:
            # Unload unused models (not accessed in last 5 minutes)
            current_time = time.time()
            unloaded_count = 0
            
            for model_name, model_info in list(self.loaded_models.items()):
                if current_time - model_info.last_accessed > 300:  # 5 minutes
                    await self.unload_model(model_name)
                    unloaded_count += 1
            
            return {
                "message": f"Memory optimization completed",
                "unloaded_models": unloaded_count,
                "current_memory_usage": self.current_memory_usage,
                "status": "optimized"
            }
            
        except Exception as e:
            logger.error(f"Error optimizing memory: {e}")
            return {"error": str(e)}

# Global model manager instance
model_manager = ModelManager()

# Convenience functions
async def list_models() -> Dict[str, Any]:
    """List all models"""
    return await model_manager.list_models()

async def load_model(model_name: str) -> Dict[str, Any]:
    """Load a model"""
    return await model_manager.load_model(model_name)

async def unload_model(model_name: str) -> Dict[str, Any]:
    """Unload a model"""
    return await model_manager.unload_model(model_name)

async def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get model info"""
    return await model_manager.get_model_info(model_name)

async def delete_model(model_name: str) -> Dict[str, Any]:
    """Delete a model"""
    return await model_manager.delete_model(model_name)
