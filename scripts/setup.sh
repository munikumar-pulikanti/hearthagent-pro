#!/bin/bash
# Setup script for hearthagent-pro local development

set -e

echo "🧠 hearthagent-pro setup"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi

echo "✓ Python $(python3 --version | awk '{print $2}')"

# Check uv
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    pip install uv
fi

echo "✓ uv installed"

# Sync dependencies
echo "📥 Syncing dependencies..."
uv sync

echo "✓ Dependencies synced"

# Create directories
mkdir -p ~/.ai-memory-vault/.chroma
mkdir -p bin

echo "✓ Directories created"

# Make scripts executable
chmod +x bin/session-log.sh

echo "✓ Scripts ready"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Start Ollama: ollama serve"
echo "  2. (Optional) Start MinIO: docker-compose up -d minio"
echo "  3. Run agent: python main.py"
echo "  4. Open web UI: flask --app app.py run --port 5555"
echo ""
echo "Pull models first:"
echo "  ollama pull llama3.2:1b"
echo "  ollama pull qwen2.5-coder:7b"
echo "  ollama pull qwen2.5-coder:14b"
echo "  ollama pull qwen2.5-coder:3b"
