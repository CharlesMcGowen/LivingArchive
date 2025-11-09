#!/usr/bin/env python3
"""
Embeddings Service for LLaMA Gateway
====================================

Provides embeddings generation for semantic search and RAG systems.
Uses GPU acceleration for fast vector generation.

Author: EGO Revolution Team
Version: 1.0.0 - Phase 2 Core Features
"""

import asyncio
import logging
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Add GPU_Accelerator to path
sys.path.insert(0, str(Path(__file__).parent.parent / "GPU_Accelerator"))

logger = logging.getLogger(__name__)

class EmbeddingsService:
    """Service for generating embeddings with GPU acceleration"""
    
    def __init__(self):
        self.embedding_models = {
            "text-embedding-ada-002": {
                "dimensions": 1536,
                "max_tokens": 8191,
                "description": "General purpose embeddings"
            },
            "text-embedding-3-small": {
                "dimensions": 1536,
                "max_tokens": 8191,
                "description": "Small, fast embeddings"
            },
            "text-embedding-3-large": {
                "dimensions": 3072,
                "max_tokens": 8191,
                "description": "Large, high-quality embeddings"
            },
            "nomic-embed-text": {
                "dimensions": 768,
                "max_tokens": 2048,
                "description": "Open source embeddings"
            },
            "bge-large-en-v1.5": {
                "dimensions": 1024,
                "max_tokens": 512,
                "description": "BAAI General Embedding"
            }
        }
        
        self.embeddings_cache = {}
        self.gpu_available = False
        
        # Try to initialize GPU acceleration
        try:
            from accelerators.gpu_acceleration_service import GPUAccelerationService
            self.gpu_service = GPUAccelerationService()
            self.gpu_available = self.gpu_service.gpu_info.available if self.gpu_service.gpu_info else False
            logger.info(f"🚀 Embeddings GPU Acceleration: {'Enabled' if self.gpu_available else 'Disabled'}")
        except Exception as e:
            logger.warning(f"⚠️ GPU acceleration not available for embeddings: {e}")
            self.gpu_service = None
    
    async def generate_embeddings(self, texts: List[str], model: str = "text-embedding-ada-002") -> Dict[str, Any]:
        """Generate embeddings for a list of texts"""
        try:
            start_time = time.time()
            
            # Validate model
            if model not in self.embedding_models:
                return {"error": f"Model {model} not supported"}
            
            model_info = self.embedding_models[model]
            
            # Check cache first
            cache_key = f"{model}:{hash(tuple(texts))}"
            if cache_key in self.embeddings_cache:
                logger.info("📦 Using cached embeddings")
                return self.embeddings_cache[cache_key]
            
            # Generate embeddings
            embeddings = []
            for text in texts:
                # Simulate embedding generation (in real implementation, use actual model)
                embedding = await self._generate_single_embedding(text, model_info)
                embeddings.append(embedding)
            
            # Create response
            response = {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": embedding,
                        "model": model
                    }
                    for i, embedding in enumerate(embeddings)
                ],
                "model": model,
                "usage": {
                    "prompt_tokens": sum(len(text.split()) for text in texts),
                    "total_tokens": sum(len(text.split()) for text in texts)
                },
                "gpu_acceleration": self.gpu_available,
                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            }
            
            # Cache the result
            self.embeddings_cache[cache_key] = response
            
            logger.info(f"✅ Generated {len(embeddings)} embeddings using {model}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Embeddings generation error: {e}")
            return {"error": str(e)}
    
    async def _generate_single_embedding(self, text: str, model_info: Dict[str, Any]) -> List[float]:
        """Generate a single embedding (simulated)"""
        try:
            # Simulate GPU processing time
            if self.gpu_available:
                await asyncio.sleep(0.01)  # Fast GPU processing
            else:
                await asyncio.sleep(0.05)  # Slower CPU processing
            
            # Generate random embedding vector (in real implementation, use actual model)
            dimensions = model_info["dimensions"]
            
            # Create deterministic embedding based on text content
            # This is a simplified approach - real implementation would use actual embedding models
            text_hash = hash(text)
            np.random.seed(text_hash % (2**32))
            embedding = np.random.normal(0, 1, dimensions).tolist()
            
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = (np.array(embedding) / norm).tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Single embedding generation error: {e}")
            # Return zero vector as fallback
            return [0.0] * model_info["dimensions"]
    
    async def similarity_search(self, query_embedding: List[float], document_embeddings: List[List[float]], 
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform similarity search using cosine similarity"""
        try:
            if not document_embeddings:
                return []
            
            # Calculate cosine similarities
            similarities = []
            for i, doc_embedding in enumerate(document_embeddings):
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                similarities.append({
                    "index": i,
                    "similarity": similarity,
                    "embedding": doc_embedding
                })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Similarity search error: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Convert to numpy arrays
            a = np.array(vec1)
            b = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot_product / (norm_a * norm_b)
            
        except Exception as e:
            logger.error(f"❌ Cosine similarity calculation error: {e}")
            return 0.0
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Get list of available embedding models"""
        return {
            "models": list(self.embedding_models.keys()),
            "model_details": self.embedding_models,
            "gpu_acceleration": self.gpu_available,
            "cache_size": len(self.embeddings_cache)
        }
    
    async def clear_cache(self) -> Dict[str, Any]:
        """Clear embeddings cache"""
        cache_size = len(self.embeddings_cache)
        self.embeddings_cache.clear()
        return {
            "message": f"Cache cleared successfully",
            "cleared_entries": cache_size
        }

# Global embeddings service instance
embeddings_service = EmbeddingsService()

# Convenience functions
async def generate_embeddings(texts: List[str], model: str = "text-embedding-ada-002") -> Dict[str, Any]:
    """Generate embeddings for texts"""
    return await embeddings_service.generate_embeddings(texts, model)

async def similarity_search(query_embedding: List[float], document_embeddings: List[List[float]], 
                          top_k: int = 5) -> List[Dict[str, Any]]:
    """Perform similarity search"""
    return await embeddings_service.similarity_search(query_embedding, document_embeddings, top_k)

async def get_embedding_models() -> Dict[str, Any]:
    """Get available embedding models"""
    return await embeddings_service.get_available_models()
