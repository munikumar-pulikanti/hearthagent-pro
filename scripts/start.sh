#!/bin/bash
# Start hearthagent-pro with all services

set -e

echo "🚀 Starting hearthagent-pro..."

# Start services
echo "Starting MinIO..."
docker-compose up -d minio

echo "Starting Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "⚠️ Ollama not found. Install from https://ollama.ai"
else
    # Check if ollama is running
    if ! curl -s http://localhost:11434/api/tags > /dev/null; then
        echo "Starting Ollama server..."
        ollama serve &
        sleep 2
    fi
fi

echo "Starting hearthagent-pro..."
echo ""

# Run main agent
python main.py
