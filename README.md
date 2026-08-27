
### Memory system (hot / warm / cold)

- Hot -- SQLite FTS5 (keyword) + ChromaDB (semantic), local, always live.
- Warm -- Turso (libSQL), shared/multi-writer, push/pull sync tested working both directions.
- Cold -- MinIO (S3-compatible), idle memories archived out of the hot tier, restored transparently on a real search miss.

New memories are embedded into ChromaDB immediately on save, no separate manual sync step required.

Confidence lifecycle: new memories start at "hypothesis". A highly similar existing memory is treated as corroboration (bumps a count, promotes confidence toward "confirmed") rather than inserted as a duplicate. A moderately similar-but-not-clearly-matching memory gets flagged "needs_review" instead of guessed at, since similarity alone can't tell agreement from conflict.

### Reliability layer

- Recursion limit -- the agent loop is capped at 15 internal steps and fails gracefully (a clear message, not a hang or a crash) if it can't converge.
- Session assertions -- deterministic, local checks run after every turn. Currently catches: claimed success without actually running a verification tool, empty responses, and unexecuted tool calls that leaked to the user as raw text instead of running.
- Faithfulness check -- flags when relevant memory was injected into context but the final answer doesn't appear to use it. Found this failure mode for real while building this: the agent ignored retrieved context and fabricated a plausible-sounding answer instead.
- Eval loop with LLM judge -- type "save eval" after confirming an answer is correct, and the task/answer/current git commit get saved as a golden baseline. bin/evals.py re-runs each baseline fresh later and has a separate model grade whether behavior still matches.

### Metrics

Every turn logs category, routed model, duration, token counts, which memory tier resolved the lookup, and any assertion flags, to a local SQLite database at ~/.ai-memory-vault/hearthagent_pro_metrics.db.

## Quick start (local mode)

Requires Ollama, uv, and Docker (for MinIO cold storage).

git clone https://github.com/munikumar-pulikanti/hearthagent-pro.git
cd hearthagent-pro
uv sync

ollama pull llama3.1:8b
ollama pull llama3.2:1b
ollama pull nomic-embed-text

docker run -d -p 9000:9000 -p 9001:9001 -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin --name hearthagent-minio minio/minio server /data --console-address ":9001"

uv run python3 main.py

## Usage

$ uv run python3 main.py
you> list the files in this project
you> remember this: FINDING. save with scope 'x', type 'note'
you> save eval
you> /quit

One-shot mode: uv run python3 main.py "your task here"

Run the eval suite: uv run python3 -m bin.evals

## Quick start (local mode)

Requires Ollama, uv, and Docker (for MinIO cold storage).

git clone https://github.com/munikumar-pulikanti/hearthagent-pro.git
cd hearthagent-pro
uv sync

ollama pull llama3.1:8b
ollama pull llama3.2:1b
ollama pull nomic-embed-text

docker run -d -p 9000:9000 -p 9001:9001 -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin --name hearthagent-minio minio/minio server /data --console-address ":9001"

uv run python3 main.py

## Usage

$ uv run python3 main.py
you> list the files in this project
you> remember this: FINDING. save with scope 'x', type 'note'
you> save eval
you> /quit

One-shot mode: uv run python3 main.py "your task here"

Run the eval suite: uv run python3 -m bin.evals
