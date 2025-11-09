#!/usr/bin/env python3
"""
LLaMA Gateway CLI - Ollama-like interface
========================================

Provides Ollama-like CLI commands for the enhanced LLaMA Gateway:
- llama pull <model>     - Pull model from various sources
- llama list             - List available models  
- llama rm <model>       - Remove model
- llama run <model>      - Run model interactively
- llama chat             - Start chat session

Usage:
    python llama_cli.py pull llama2:7b
    python llama_cli.py list
    python llama_cli.py rm llama2:7b
    python llama_cli.py run llama2:7b
    python llama_cli.py chat
"""

import sys
import json
import asyncio
import aiohttp
import argparse
from typing import Optional, Dict, Any

class LLaMACLI:
    """CLI interface for LLaMA Gateway"""
    
    def __init__(self, gateway_url: str = "http://localhost:8082"):
        self.gateway_url = gateway_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def pull_model(self, model_name: str, source: str = "huggingface", quantize: bool = True):
        """Pull model from various sources"""
        try:
            async with self.session.post(
                f"{self.gateway_url}/models/pull",
                json={
                    "name": model_name,
                    "source": source,
                    "quantize": quantize
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Successfully pulled {model_name}")
                    print(f"   Source: {result.get('source', 'unknown')}")
                    print(f"   Path: {result.get('path', 'unknown')}")
                    print(f"   Quantized: {result.get('quantized', False)}")
                    return True
                else:
                    error = await response.text()
                    print(f"❌ Failed to pull {model_name}: {error}")
                    return False
        except Exception as e:
            print(f"❌ Error pulling model: {e}")
            return False
    
    async def list_models(self):
        """List available models"""
        try:
            async with self.session.get(f"{self.gateway_url}/api/models") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("models", [])
                    total = data.get("total", len(models))
                    
                    if not models and total == 0:
                        print("📦 No models currently loaded")
                        print("\nTo load a model, use:")
                        print("  python llama_cli.py pull <model_name>")
                        return
                    
                    print(f"Available models ({total}):")
                    print("-" * 60)
                    for model in models:
                        if isinstance(model, dict):
                            name = model.get("name", "unknown")
                            source = model.get("source", "unknown")
                            quantized = model.get("quantized", False)
                            downloaded = model.get("downloaded_at", "unknown")
                            
                            print(f"📦 {name}")
                            print(f"   Source: {source}")
                            print(f"   Quantized: {'Yes' if quantized else 'No'}")
                            print(f"   Downloaded: {downloaded}")
                            print()
                        else:
                            print(f"📦 {model}")
                else:
                    print(f"❌ Failed to list models: HTTP {response.status}")
                    error_text = await response.text()
                    if error_text:
                        print(f"   Error: {error_text[:200]}")
        except Exception as e:
            print(f"❌ Error listing models: {e}")
    
    async def remove_model(self, model_name: str):
        """Remove model"""
        try:
            async with self.session.delete(f"{self.gateway_url}/models/{model_name}") as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Successfully removed {model_name}")
                    return True
                else:
                    error = await response.text()
                    print(f"❌ Failed to remove {model_name}: {error}")
                    return False
        except Exception as e:
            print(f"❌ Error removing model: {e}")
            return False
    
    async def run_model(self, model_name: str, prompt: str):
        """Run model with prompt"""
        try:
            # First load the model
            async with self.session.post(
                f"{self.gateway_url}/models/load",
                json={"model_id": model_name}
            ) as response:
                if response.status != 200:
                    print(f"❌ Failed to load model {model_name}")
                    return False
            
            # Generate response
            async with self.session.post(
                f"{self.gateway_url}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": 512,
                    "temperature": 0.7
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"🤖 {model_name}:")
                    print(result.get("text", "No response"))
                    return True
                else:
                    error = await response.text()
                    print(f"❌ Failed to generate: {error}")
                    return False
        except Exception as e:
            print(f"❌ Error running model: {e}")
            return False
    
    async def chat_session(self, model_name: str = "llama-custom"):
        """Start interactive chat session"""
        try:
            print(f"🤖 Starting chat with {model_name}")
            print("Type 'quit' to exit, 'clear' to clear context")
            print("-" * 50)
            
            while True:
                try:
                    user_input = input("You: ").strip()
                    
                    if user_input.lower() == 'quit':
                        print("👋 Goodbye!")
                        break
                    elif user_input.lower() == 'clear':
                        print("🧹 Context cleared")
                        continue
                    elif not user_input:
                        continue
                    
                    # Send chat request
                    async with self.session.post(
                        f"{self.gateway_url}/v1/chat/completions",
                        json={
                            "model": model_name,
                            "messages": [
                                {"role": "user", "content": user_input}
                            ],
                            "max_tokens": 512,
                            "temperature": 0.7
                        }
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            ai_response = result.get("choices", [{}])[0].get("message", {}).get("content", "No response")
                            print(f"AI: {ai_response}")
                        else:
                            error = await response.text()
                            print(f"❌ Error: {error}")
                            
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    
        except Exception as e:
            print(f"❌ Error starting chat: {e}")

async def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="LLaMA Gateway CLI")
    parser.add_argument("command", choices=["pull", "list", "rm", "run", "chat"], 
                       help="Command to execute")
    parser.add_argument("model", nargs="?", help="Model name")
    parser.add_argument("--source", default="huggingface", 
                       choices=["huggingface", "ollama", "local"],
                       help="Source for pulling models")
    parser.add_argument("--no-quantize", action="store_true",
                       help="Don't quantize model")
    parser.add_argument("--prompt", help="Prompt for run command")
    parser.add_argument("--gateway-url", default="http://localhost:8082",
                       help="LLaMA Gateway URL")
    
    args = parser.parse_args()
    
    async with LLaMACLI(args.gateway_url) as cli:
        if args.command == "pull":
            if not args.model:
                print("❌ Model name required for pull command")
                sys.exit(1)
            await cli.pull_model(args.model, args.source, not args.no_quantize)
            
        elif args.command == "list":
            await cli.list_models()
            
        elif args.command == "rm":
            if not args.model:
                print("❌ Model name required for rm command")
                sys.exit(1)
            await cli.remove_model(args.model)
            
        elif args.command == "run":
            if not args.model:
                print("❌ Model name required for run command")
                sys.exit(1)
            prompt = args.prompt or input("Enter prompt: ")
            await cli.run_model(args.model, prompt)
            
        elif args.command == "chat":
            model = args.model or "llama-custom"
            await cli.chat_session(model)

if __name__ == "__main__":
    asyncio.run(main())

