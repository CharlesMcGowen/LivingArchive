#!/usr/bin/env python3
"""
Simple Model Fallback Setup
============================

Creates a simple mock model provider when full GPU infrastructure isn't available.
This allows the gateway to give basic responses while the full GPU setup is configured.

Author: EGO Revolution Team
"""

import sys
from pathlib import Path

# Add EgoLlama to path
sys.path.insert(0, str(Path(__file__).parent))

# Create a simple fallback GPU module
fallback_module_code = '''#!/usr/bin/env python3
"""
Simple Fallback GPU Module
==========================

Provides basic model functionality when full GPU infrastructure isn't available.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SimpleFallbackModel:
    """Simple fallback model using rule-based responses"""
    
    def __init__(self):
        self.model_loaded = False
        self.model_name = "simple-fallback"
        logger.info("✅ Simple fallback model initialized")
    
    def load_model(self, model_id: str = "simple-fallback") -> bool:
        """Load the fallback model"""
        self.model_loaded = True
        self.model_name = model_id
        logger.info(f"✅ Fallback model '{model_id}' loaded")
        return True
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> Dict[str, Any]:
        """Generate a simple response"""
        if not self.model_loaded:
            return {
                'error': 'Model not loaded',
                'generated_text': 'Please load a model first using /models/load endpoint'
            }
        
        # Simple template-based response
        response_parts = []
        
        # Check for system architecture questions
        if any(keyword in prompt.lower() for keyword in ['system', 'architecture', 'how do you work']):
            response_parts.append("""I'm an AI assistant running in a Docker container! 🎉

My system architecture includes:
- LLaMA Gateway for AI inference (currently using fallback mode)
- PostgreSQL database for conversation history
- WebSocket server for real-time chat
- Code executor for PowerShell and Bash scripts
- Web search capabilities for research

I'm currently in fallback mode, but I can still help with code, debugging, and general questions! 💖""")
        
        # Check for database questions
        elif any(keyword in prompt.lower() for keyword in ['database', 'postgresql', 'sqlalchemy', 'store']):
            response_parts.append("""I use PostgreSQL for persistent storage! 💾

- Database: postgresql://postgres:postgres@localhost:5432/ego
- ORM: SQLAlchemy (ORM + Core style)
- Tables: Chat conversations, messages, and execution history

I store conversation history, code execution results, and can remember our chats across sessions! 🎉""")
        
        # Check for code execution questions
        elif any(keyword in prompt.lower() for keyword in ['execute', 'code', 'shell', 'powershell', 'bash']):
            response_parts.append("""I can execute code! 💻

Supported shells:
- Bash: ✅ Available
- PowerShell: Available on Windows or with pwsh

I can:
- Run scripts and commands
- Debug output and errors
- Store execution history in the database

Just ask me to execute something and I'll run it for you! 🎉""")
        
        # Check for web search questions
        elif any(keyword in prompt.lower() for keyword in ['search', 'web', 'rag', 'research']):
            response_parts.append("""I have web search capabilities! 🔍

I can:
- Search DuckDuckGo and Google
- Scrape web pages
- Query NVD CVE database
- Use RAG for research lookups

This helps me find vulnerability information, research papers, and other resources! 🎉""")
        
        # Default response
        else:
            response_parts.append(f"""I'm your AI coding assistant! 🎉

I can help with:
- Code debugging and optimization
- PowerShell and Bash scripts
- Docker and containerization
- System architecture questions
- Bug bounty research coordination

I'm currently in fallback mode, but I'm ready to help! 💖

(Note: To enable full AI responses, load a LLaMA model in the gateway)""")
        
        generated_text = response_parts[0] if response_parts else "I'm here to help! 💖"
        
        return {
            'generated_text': generated_text,
            'prompt_tokens': len(prompt.split()),
            'completion_tokens': len(generated_text.split()),
            'total_tokens': len(prompt.split()) + len(generated_text.split()),
            'inference_time_ms': 50,  # Fast fallback
            'tokens_per_second': 10.0,
            'model': self.model_name
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        return {
            'model_loaded': self.model_loaded,
            'model_name': self.model_name,
            'gpu_available': False,
            'device': 'fallback',
            'quantization': 'None',
            'total_inferences': 0,
            'average_tokens_per_second': 10.0
        }
    
    def unload_model(self) -> bool:
        """Unload model"""
        self.model_loaded = False
        logger.info("Model unloaded")
        return True

# Global instance
_fallback_model = None

def get_fallback_model():
    """Get fallback model instance"""
    global _fallback_model
    if _fallback_model is None:
        _fallback_model = SimpleFallbackModel()
    return _fallback_model

# Export as GPU module interface
class GPUModule:
    """GPU Module wrapper for fallback"""
    
    def __init__(self):
        self.model = get_fallback_model()
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> Dict[str, Any]:
        """Generate text"""
        return self.model.generate(prompt, max_tokens, temperature)
    
    def load_model(self, model_id: str, quantization_bits: int = 4) -> bool:
        """Load model"""
        return self.model.load_model(model_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats"""
        return self.model.get_stats()
    
    def unload_model(self) -> bool:
        """Unload model"""
        return self.model.unload_model()

def get_gpu_module():
    """Get GPU module (fallback implementation)"""
    return GPUModule()
'''

# Write fallback module
fallback_path = Path(__file__).parent / "gpu_fallback.py"
with open(fallback_path, 'w') as f:
    f.write(fallback_module_code)

print(f"✅ Created fallback GPU module at: {fallback_path}")
print("\nTo use in gateway:")
print("  1. Update simple_llama_gateway_crash_safe.py to import from gpu_fallback")
print("  2. Or set environment variable: EGOLLAMA_USE_FALLBACK=true")

