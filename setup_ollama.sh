#!/bin/bash
"""
Ollama Setup Script for LLaMA Gateway
====================================

This script helps set up Ollama for the llama-gateway system.
It installs Ollama, pulls recommended models, and configures the system.

Usage: ./setup_ollama.sh
"""

echo "🚀 Setting up Ollama for LLaMA Gateway..."

# Check if Ollama is already installed
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is already installed"
    ollama --version
else
    echo "📥 Installing Ollama..."
    
    # Install Ollama
    curl -fsSL https://ollama.ai/install.sh | sh
    
    # Wait for installation to complete
    sleep 5
    
    # Check if installation was successful
    if command -v ollama &> /dev/null; then
        echo "✅ Ollama installed successfully"
    else
        echo "❌ Ollama installation failed"
        echo "Please install Ollama manually: https://ollama.ai/download"
        exit 1
    fi
fi

# Start Ollama service
echo "🔄 Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to start
echo "⏳ Waiting for Ollama to start..."
sleep 10

# Check if Ollama is running
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama service is running"
else
    echo "❌ Ollama service failed to start"
    exit 1
fi

# Pull recommended models
echo "📦 Pulling recommended models..."

# Small, fast models for testing
echo "📥 Pulling llama3.2:1b (small, fast model)..."
ollama pull llama3.2:1b

echo "📥 Pulling mistral:7b (good balance)..."
ollama pull mistral:7b

echo "📥 Pulling phi3:3.8b (Microsoft's model)..."
ollama pull phi3:3.8b

# Coding models
echo "📥 Pulling codellama:7b (coding specialist)..."
ollama pull codellama:7b

# List available models
echo "📋 Available models:"
ollama list

echo "🎉 Ollama setup complete!"
echo ""
echo "Available endpoints:"
echo "  - Health: http://localhost:8088/health"
echo "  - Ollama Status: http://localhost:8088/ollama/status"
echo "  - Models: http://localhost:8088/models"
echo "  - Chat: http://localhost:8088/v1/chat/completions"
echo ""
echo "To test the integration:"
echo "  curl -s http://localhost:8088/ollama/status"
echo ""
echo "To stop Ollama: kill $OLLAMA_PID"
