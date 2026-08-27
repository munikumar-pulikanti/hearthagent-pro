"""Tools for hearthagent-pro -- file I/O, shell, code search, memory (FTS + semantic RAG), web search.
Shares the same ~/.ai-memory-vault/ data as the hearthagent baseline -- same schema, same files."""
import subprocess
import sqlite3
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from ddgs import DDGS

PROJECT_ROOT = Path.cwd()
ALLOWED_SHELL_COMMANDS = {"ls", "cat", "pwd", "grep", "find", "python", "python3", "pytest", "git", "uv"}
VAULT_DB = Path.home() / ".ai-memory-vault" / "global_brain.db"
CHROMA_PATH = Path.home() / ".ai-memory-vault" / "chroma"
COLLECTION_NAME = "memories"

_embedder = None
_chroma_client = None


def read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    try:
        return p.read_text()
    except Exception as e:
        return f"ERROR reading {path}: {e}"


def write_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content)} chars to {path}"


def list_dir(path: str = ".") -> str:
    p = Path(path)
    if not p.exists():
        return f"ERROR: path not found: {path}"
    entries = sorted(p.iterdir())
    return "\n".join(f"{'d' if e.is_dir() else 'f'} {e.name}" for e in entries)


def search_code(query: str, path: str = ".") -> str:
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", query, path],
            capture_output=True, text=True, timeout=15
        )
        return result.stdout[:3000] or "No matches found."
    except Exception as e:
        return f"ERROR searching: {e}"


def run_shell(command: str) -> str:
    first_word = command.strip().split()[0] if command.strip() else ""
    if first_word not in ALLOWED_SHELL_COMMANDS:
        return f"REJECTED: '{first_word}' is not allowlisted ({sorted(ALLOWED_SHELL_COMMANDS)})"
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=30, cwd=PROJECT_ROOT
        )
        output = result.stdout + result.stderr
        return output[:3000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out"
    except Exception as e:
        return f"ERROR running command: {e}"


def memory_search(query: str, limit: int = 5) -> str:
    if not VAULT_DB.exists():
        return "No memory vault found at ~/.ai-memory-vault/"
    conn = sqlite3.connect(VAULT_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT m.scope, m.type, m.content FROM memories_fts f "
            "JOIN memories m ON m.id = f.rowid "
            "WHERE f.memories_fts MATCH ? LIMIT ?",
            (query, limit)
        ).fetchall()
    except Exception as e:
        return f"ERROR searching memory: {e}"
    finally:
        conn.close()
    if not rows:
        return "No matching memories."
    return "\n".join(f"[{r['scope']}/{r['type']}] {r['content']}" for r in rows)


def memory_save(scope: str, type_: str, content: str, tags: str = "") -> str:
    VAULT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(VAULT_DB)
    conn.execute(
        "INSERT INTO memories (scope, type, tags, content, updated_at) "
        "VALUES (?, ?, ?, ?, unixepoch())",
        (scope, type_, tags, content)
    )
    conn.commit()
    conn.close()
    return f"Saved memory to scope={scope}, type={type_}"


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_chroma_collection():
    global _chroma_client
    if _chroma_client is None:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return _chroma_client.get_or_create_collection(COLLECTION_NAME)


def memory_sync_embeddings() -> str:
    if not VAULT_DB.exists():
        return "No memory vault found."
    conn = sqlite3.connect(VAULT_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, scope, type, content FROM memories").fetchall()
    conn.close()

    if not rows:
        return "No memory rows to sync."

    embedder = _get_embedder()
    collection = _get_chroma_collection()

    ids = [str(r["id"]) for r in rows]
    documents = [r["content"] for r in rows]
    metadatas = [{"scope": r["scope"], "type": r["type"]} for r in rows]
    embeddings = embedder.encode(documents).tolist()

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return f"Synced {len(rows)} memories into ChromaDB."


def memory_semantic_search(query: str, top: int = 5) -> str:
    collection = _get_chroma_collection()
    embedder = _get_embedder()
    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=top)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        return "No semantic matches found. Try memory_sync_embeddings_tool first if memory is non-empty."

    lines = [f"[{m['scope']}/{m['type']}] {d}" for d, m in zip(docs, metas)]
    return "\n".join(lines)


def web_search(query: str, max_results: int = 5) -> str:
    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as e:
        return f"ERROR searching web: {e}"
    if not results:
        return "No results found."
    lines = []
    for r in results:
        lines.append(f"- {r.get('title', '')}\n  {r.get('href', '')}\n  {r.get('body', '')[:200]}")
    return "\n".join(lines)
