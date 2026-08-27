# hearthagent-pro — Roadmap

Local-first AI agent with production-grade memory, model routing, evals, and cold storage.
Zero cloud dependency in local mode. Pluggable backends to scale to AWS when needed.

---

## What exists today (hearthagent baseline)

- LangGraph ReAct agent on Ollama (`qwen3:8b` / `llama3.1:8b`)
- Real tool-calling: `read_file`, `write_file`, `list_dir`, `search_code`, `run_shell`
- Persistent memory: SQLite FTS5 + ChromaDB (`all-MiniLM-L6-v2`, fully local)
- `mem update` trigger phrase → saves confirmed findings back to memory
- DuckDuckGo web search fallback (no API key)
- `uv` for dependency management

---

## What to add (the pro layer)

### 1. Model Router
**File:** `agent/router.py`

Classify task type in one cheap call, route to the right model.

```python
ROUTING_TABLE = {
    "investigate":   "qwen2.5-coder:14b",   # deep reasoning — use the big model
    "implement":     "qwen2.5-coder:7b",     # balanced
    "unit_tests":    "qwen2.5-coder:3b",     # cheap, pattern-heavy
    "extract":       "llama3.2:1b",          # classify/route — fastest local model
}
```

Classifier: one call to `llama3.2:1b` with a tight prompt — ~100ms, negligible cost.
On AWS: swap classifier for Nova Micro ($0.035/MTok). Same routing table.

**Why it matters:** `qwen2.5-coder:14b` on CPU is 3–5 min per response.
A unit test job routed to `qwen2.5-coder:3b` finishes in 30s. Same quality for that task.

---

### 2. Memory Tiers — Hot / Warm / Cold

**Current:** single SQLite + ChromaDB (hot only).

**Add:**

| Tier | Local | Cloud-agnostic | AWS | GCP | Azure |
|------|-------|----------------|-----|-----|-------|
| Hot  | SQLite FTS5 + ChromaDB | same | same | same | same |
| Warm | `shared_brain.db` (Turso in prod) | Turso (libSQL) / MongoDB | DynamoDB | Firestore | Cosmos DB |
| Cold | MinIO (Docker) | MinIO / Backblaze B2 / Cloudflare R2 | S3 | GCS | Azure Blob |

**File:** `bin/cold_storage.py`

```bash
python3 bin/cold_storage.py archive --dry-run   # preview candidates
python3 bin/cold_storage.py archive --days 90   # snapshot to cold tier
python3 bin/cold_storage.py restore <row_id>    # pull back on demand
```

---

### 3. Evals
**File:** `bin/savings_tracker.py`

Track retrieval quality per task. Logged to `~/.ai-memory-vault/savings_tracker.db`.

```bash
python3 bin/savings_tracker.py log \
  --type investigate \
  --hits 2 \
  --source local-memory \
  --quality good_hit \
  --tool-calls 4 \
  --desc "root cause traced via memory hit"
```

---

### 4. Web UI
**File:** `app.py` (Flask)

```
GET  /                    → memory browser
GET  /api/memories        → list rows (filter: scope, type, confidence, q)
POST /api/corroborate/<id>
POST /api/cite/<id>
POST /api/flag-review/<id>
GET  /api/stats           → counts by scope/confidence
GET  /api/cold-storage    → archive stats + warmup events
GET  /api/savings         → task savings report
```

Local only — binds to `127.0.0.1:5555`. No auth needed.

---

### 5. Pluggable Backends via config.yaml

Single file controls everything. One env var to switch: `HEARTHAGENT_MODE=prod python main.py`.
