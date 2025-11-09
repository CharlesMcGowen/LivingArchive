#!/bin/bash
# Install LLaMA CLI alias for Linux/macOS
# ========================================

echo "🔧 Installing LLaMA CLI alias..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_SCRIPT="$SCRIPT_DIR/llama"

# Check if the llama script exists
if [ ! -f "$LLAMA_SCRIPT" ]; then
    echo "❌ LLaMA script not found at $LLAMA_SCRIPT"
    exit 1
fi

# Make sure it's executable
chmod +x "$LLAMA_SCRIPT"

# Create symlink in /usr/local/bin (requires sudo)
echo "📦 Creating system-wide alias..."
if command -v sudo >/dev/null 2>&1; then
    sudo ln -sf "$LLAMA_SCRIPT" /usr/local/bin/llama
    echo "✅ LLaMA CLI installed to /usr/local/bin/llama"
else
    echo "⚠️  Sudo not available. Creating local alias instead..."
    echo "Add this to your ~/.bashrc or ~/.zshrc:"
    echo "alias llama='$LLAMA_SCRIPT'"
    echo ""
    echo "Then run: source ~/.bashrc (or ~/.zshrc)"
fi

# Test the installation
echo "🧪 Testing installation..."
if command -v llama >/dev/null 2>&1; then
    echo "✅ LLaMA CLI is now available system-wide!"
    echo ""
    echo "Usage examples:"
    echo "  llama pull llama2:7b"
    echo "  llama list"
    echo "  llama serve"
    echo "  llama status"
else
    echo "⚠️  LLaMA CLI not found in PATH. Please add to your shell profile:"
    echo "export PATH=\"$SCRIPT_DIR:\$PATH\""
fi

echo ""
echo "🎉 Installation complete!"
echo "💡 Run 'llama' to see all available commands"

