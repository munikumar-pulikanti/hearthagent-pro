# Troubleshooting

## Common Issues

### "Ollama connection refused"

Ollama server is not running.

```bash
ollama serve
```

Then in another terminal:
```bash
ollama pull llama3.2:1b  # If needed
python main.py
```

### "MinIO connection refused"

MinIO is not running. Start it:

```bash
docker-compose up -d minio
```

Verify:
```bash
curl http://localhost:9000/minio/health/live
```

### "config.yaml not found"

Make sure you're running from the repo root:

```bash
cd hearthagent-pro
python main.py
```

### "Model not found" error

Pull the model first:

```bash
ollama pull qwen2.5-coder:7b
```

Check available models:
```bash
ollama list
```

### Memory database corrupted

Reset and recreate:

```bash
rm ~/.ai-memory-vault/global_brain.db
python main.py  # Will recreate schema
```

### "cold_storage.py: bucket not found"

Create the MinIO bucket:

```bash
mc mb minio/hearthagent-cold
```

Or via MinIO Console (http://localhost:9001):
1. Log in (minioadmin / minioadmin)
2. Click "Buckets" → "Create Bucket"
3. Name: `hearthagent-cold`

### Web UI won't start

Check if port 5555 is in use:

```bash
lsof -i :5555
```

Kill the process or use a different port:

```bash
flask --app app.py run --port 5556
```

## Debug Mode

Enable debug logging:

```bash
HEARTHAGENT_DEBUG=1 python main.py
```

Or in your shell:
```bash
export HEARTHAGENT_DEBUG=1
python main.py
```

## Performance Issues

### Agent is slow

1. **Classifier too slow:** Use a smaller model
   ```yaml
   router:
     local:
       classifier_model: llama3.2:1b  # Already the smallest
   ```

2. **Routing model too slow:** Reduce model size
   ```yaml
   routing_table:
     unit_tests: qwen2.5-coder:1b  # Smaller variant
   ```

3. **Memory queries slow:** Enable cold storage archival
   ```bash
   python bin/cold_storage.py archive --days 90
   ```

### Memory usage high

1. Archive old memories:
   ```bash
   python bin/cold_storage.py archive --days 30
   ```

2. Reduce embeddings model size (if switching to prod):
   ```yaml
   embeddings:
     prod:
       model: all-MiniLM-L6-v2  # Smaller than titan
   ```

## Getting Help

- **Issues:** https://github.com/munikumar-pulikanti/hearthagent-pro/issues
- **Discussions:** https://github.com/munikumar-pulikanti/hearthagent-pro/discussions
- **Baseline:** https://github.com/munikumar-pulikanti/hearthagent
