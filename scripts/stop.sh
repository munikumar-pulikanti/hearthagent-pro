#!/bin/bash
# Stop all services

echo "⏹️ Stopping hearthagent-pro services..."

echo "Stopping MinIO..."
docker-compose down --remove-orphans

echo "Stopping Ollama..."
pkill -f "ollama serve" || true

echo "✅ All services stopped"
