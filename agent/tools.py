"""Tools for hearthagent-pro -- file I/O, shell, code search, memory
(FTS + semantic RAG with confidence lifecycle + hot/warm/cold cascade), web search."""
import json
import os
import subprocess
import sqlite3
from pathlib import Path

import boto3
import chromadb
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from ddgs import DDGS
import requests
import re
from agent import metrics
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path.cwd()
ALLOWED_SHELL_COMMANDS = {"ls", "cat", "pwd", "grep", "find", "python", "python3", "pytest", "git", "uv"}
VAULT_DB = Path.home() / ".ai-memory-vault" / "global_brain.db"
CHROMA_PATH = Path.home() / ".ai-memory-vault" / "chroma"
COLLECTION_NAME = "memories"
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

TURSO_URL = os.environ.get("TURSO_URL")
TURSO_AUTH_TOKEN = os.environ.get("TURSO_AUTH_TOKEN")

# Cascade thresholds -- similarity (1 = identical, 0 = unrelated).
HOT_ACCEPT_THRESHOLD = 0.4     # your own data, lowest bar
WARM_ACCEPT_THRESHOLD = 0.65   # shared/other-people's data, higher bar
COLD_ACCEPT_THRESHOLD = 0.5    # archived, last resort

# Confidence lifecycle thresholds, on save.
CORROBORATION_THRESHOLD = 0.85  # "this is the same finding restated"
REVIEW_BAND_LOW = 0.5           # below this: unrelated, just insert fresh
REVIEW_BAND_HIGH = 0.85         # between LOW and CORROBORATION: related but
                                 # unverified as same-or-conflicting -> flag

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


SEARCH_EXCLUDE_DIRS = [".venv", "venv", ".git", "__pycache__", "node_modules", ".pytest_cache"]


def search_code(query: str, path: str = ".") -> str:
    try:
        cmd = ["grep", "-rn", "--include=*.py"]
        for d in SEARCH_EXCLUDE_DIRS:
            cmd += ["--exclude-dir=" + d]
        cmd += [query, path]
        result = subprocess.run(
            cmd,
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


def _ensure_confidence_columns():
    conn = sqlite3.connect(VAULT_DB)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
    if "corroboration_count" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN corroboration_count INTEGER DEFAULT 1")
    if "needs_review" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN needs_review INTEGER DEFAULT 0")
    if "evidence_url" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN evidence_url TEXT")
    if "evidence_verified" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN evidence_verified INTEGER DEFAULT 0")
    conn.commit()
    conn.close()


def _verify_evidence_url(url: str) -> bool:
    """Check that an evidence URL actually resolves. A citation that 404s
    is not evidence -- this mirrors requiring a real Jira/AWS/code
    reference, not just a claim that one exists."""
    if not url:
        return False
    try:
        resp = requests.head(url, timeout=5, allow_redirects=True)
        return resp.status_code < 400
    except Exception:
        try:
            resp = requests.get(url, timeout=5)
            return resp.status_code < 400
        except Exception:
            return False


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
    return _chroma_client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def memory_search(query: str, limit: int = 5) -> str:
    if not VAULT_DB.exists():
        return "No memory vault found at ~/.ai-memory-vault/"
    conn = sqlite3.connect(VAULT_DB)
    conn.row_factory = sqlite3.Row
    # FTS5 treats :, -, *, AND/OR/NOT as query syntax. Quoting the whole
    # query as a literal phrase avoids syntax errors on ordinary input
    # like "hearthagent-pro" or "remember this:".
    safe_query = '"' + query.replace('"', '""') + '"'
    try:
        rows = conn.execute(
            "SELECT m.scope, m.type, m.content, m.confidence FROM memories_fts f "
            "JOIN memories m ON m.id = f.rowid "
            "WHERE f.memories_fts MATCH ? AND COALESCE(m.archived, 0) = 0 LIMIT ?",
            (safe_query, limit)
        ).fetchall()
    except Exception as e:
        return f"ERROR searching memory: {e}"
    finally:
        conn.close()
    if not rows:
        return "No matching memories."
    return "\n".join(f"[{r['scope']}/{r['type']}, {r['confidence']}] {r['content']}" for r in rows)


def _find_most_similar_active(content: str):
    """Find the single most similar existing active memory, for corroboration
    and conflict-review checks at save time. Returns (id, similarity) or None."""
    collection = _get_chroma_collection()
    embedder = _get_embedder()
    query_vec = embedder.encode([content]).tolist()
    results = collection.query(query_embeddings=query_vec, n_results=1, include=["distances"])
    dists = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]
    if not dists:
        return None
    similarity = 1 - dists[0]
    return (int(ids[0]), similarity)


def memory_save(scope: str, type_: str, content: str, tags: str = "", evidence_url: str = "") -> str:
    """Save a finding. New memories start at 'hypothesis'. A highly similar
    existing memory is treated as corroboration (bump count, promote
    confidence) instead of a duplicate insert. A moderately similar memory
    is flagged for review rather than assumed to agree or conflict --
    embedding similarity alone can't tell those apart.

    Confidence is gated on evidence: without a verified evidence_url
    (a real link that actually resolves -- a commit, a ticket, a doc),
    confidence caps at 'suspected' regardless of corroboration count.
    Restating an unverified claim many times does not make it confirmed."""
    _ensure_confidence_columns()
    VAULT_DB.parent.mkdir(parents=True, exist_ok=True)

    evidence_verified = 1 if evidence_url and _verify_evidence_url(evidence_url) else 0

    match = _find_most_similar_active(content)

    if match and match[1] >= CORROBORATION_THRESHOLD:
        existing_id, sim = match
        conn = sqlite3.connect(VAULT_DB)
        row = conn.execute(
            "SELECT corroboration_count, evidence_verified FROM memories WHERE id = ?", (existing_id,)
        ).fetchone()
        new_count = (row[0] or 1) + 1
        existing_evidence_verified = row[1] if row and len(row) > 1 else 0
        has_evidence = evidence_verified or existing_evidence_verified

        if has_evidence:
            new_confidence = "confirmed" if new_count >= 3 else "suspected"
        else:
            new_confidence = "suspected"  # capped -- no verified evidence, ever

        update_evidence = ", evidence_url = ?, evidence_verified = 1" if evidence_verified and not existing_evidence_verified else ""
        params = [new_count, new_confidence]
        if update_evidence:
            params.append(evidence_url)
        params.append(existing_id)

        conn.execute(
            f"UPDATE memories SET corroboration_count = ?, confidence = ?, "
            f"updated_at = unixepoch(){update_evidence} WHERE id = ?",
            params
        )
        conn.commit()
        conn.close()
        evidence_note = " (evidence-backed)" if has_evidence else " (no verified evidence -- capped at suspected)"
        return (f"Corroborated existing memory {existing_id} (similarity {sim:.2f}) -- "
                f"now confidence={new_confidence}, corroboration_count={new_count}{evidence_note}")

    conn = sqlite3.connect(VAULT_DB)
    needs_review = 1 if match and REVIEW_BAND_LOW <= match[1] < REVIEW_BAND_HIGH else 0
    cursor = conn.execute(
        "INSERT INTO memories (scope, type, tags, content, confidence, corroboration_count, "
        "needs_review, evidence_url, evidence_verified, updated_at) "
        "VALUES (?, ?, ?, ?, 'hypothesis', 1, ?, ?, ?, unixepoch())",
        (scope, type_, tags, content, needs_review, evidence_url or None, evidence_verified)
    )
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # Immediately embed the new memory so it's findable by semantic search
    # AND by future corroboration checks right away -- without this, every
    # save is invisible to _find_most_similar_active until a manual sync.
    try:
        embedder = _get_embedder()
        collection = _get_chroma_collection()
        collection.upsert(
            ids=[str(new_id)], documents=[content],
            metadatas=[{"scope": scope, "type": type_}],
            embeddings=embedder.encode([content]).tolist(),
        )
    except Exception as e:
        print(f"Warning: memory {new_id} saved but not embedded: {e}")

    if needs_review:
        return (f"Saved memory {new_id} as hypothesis, but flagged needs_review=1 -- "
                f"similar to memory {match[0]} (similarity {match[1]:.2f}), unclear if it "
                f"agrees or conflicts. Manual review recommended.")
    evidence_note = " (evidence verified)" if evidence_verified else " (no evidence -- can never exceed suspected without one)"
    return f"Saved memory {new_id} to scope={scope}, type={type_}, confidence=hypothesis{evidence_note}"


def memory_sync_embeddings() -> str:
    if not VAULT_DB.exists():
        return "No memory vault found."
    conn = sqlite3.connect(VAULT_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, scope, type, content FROM memories WHERE COALESCE(archived, 0) = 0"
    ).fetchall()
    conn.close()

    if not rows:
        return "No active memory rows to sync."

    embedder = _get_embedder()
    collection = _get_chroma_collection()

    ids = [str(r["id"]) for r in rows]
    documents = [r["content"] for r in rows]
    metadatas = [{"scope": r["scope"], "type": r["type"]} for r in rows]
    embeddings = embedder.encode(documents).tolist()

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return f"Synced {len(rows)} active memories into ChromaDB."


def _load_cold_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["cold_storage"][cfg.get("mode", "local")]


def _warm_semantic_scan(query: str, limit: int = 3):
    """Scan Turso (warm tier) for matches when hot tier misses. Returns
    list of dicts with a higher trust bar than hot, since this is
    shared/less-curated data."""
    if not TURSO_URL or not TURSO_AUTH_TOKEN:
        return []
    try:
        import libsql_experimental as libsql
        conn = libsql.connect("warm-scan-replica.db", sync_url=TURSO_URL, auth_token=TURSO_AUTH_TOKEN)
        conn.sync()
        rows = conn.execute("SELECT id, scope, type, content FROM memories").fetchall()
    except Exception:
        return []

    if not rows:
        return []

    embedder = _get_embedder()
    query_vec = embedder.encode([query])[0]

    scored = []
    for row in rows:
        row_id, scope, type_, content = row[0], row[1], row[3], row[5] if len(row) > 5 else row[-1]
        doc_vec = embedder.encode([content])[0]
        sim = float(np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-9))
        scored.append((sim, {"id": row_id, "scope": scope, "type": type_, "content": content}))

    scored.sort(key=lambda x: -x[0])
    return [d for s, d in scored[:limit] if s >= WARM_ACCEPT_THRESHOLD]


def _cold_warmup_scan(query: str, limit: int = 3):
    cfg = _load_cold_config()
    client = boto3.client(
        "s3", endpoint_url=cfg.get("endpoint") or None,
        aws_access_key_id=cfg.get("access_key", ""),
        aws_secret_access_key=cfg.get("secret_key", ""),
    )
    try:
        listing = client.list_objects_v2(Bucket=cfg["bucket"], Prefix="memories/")
    except Exception:
        return []

    objects = listing.get("Contents", [])
    if not objects:
        return []

    embedder = _get_embedder()
    query_vec = embedder.encode([query])[0]

    scored = []
    for obj in objects[:50]:
        try:
            data = json.loads(client.get_object(Bucket=cfg["bucket"], Key=obj["Key"])["Body"].read())
        except Exception:
            continue
        doc_vec = embedder.encode([data["content"]])[0]
        sim = float(np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-9))
        scored.append((sim, data))

    scored.sort(key=lambda x: -x[0])
    return [d for s, d in scored[:limit] if s >= COLD_ACCEPT_THRESHOLD]


def _restore_from_cold(data: dict):
    conn = sqlite3.connect(VAULT_DB)
    conn.execute("UPDATE memories SET archived = 0 WHERE id = ?", (data["id"],))
    conn.commit()
    conn.close()
    embedder = _get_embedder()
    collection = _get_chroma_collection()
    collection.upsert(
        ids=[str(data["id"])], documents=[data["content"]],
        metadatas=[{"scope": data["scope"], "type": data["type"]}],
        embeddings=embedder.encode([data["content"]]).tolist(),
    )


def memory_semantic_search(query: str, top: int = 5) -> str:
    """Cascading semantic search: hot tier (bar 0.4) -> warm/Turso (bar 0.65)
    -> cold/MinIO (bar 0.5), transparently restoring anything found back
    into the hot tier."""
    collection = _get_chroma_collection()
    embedder = _get_embedder()
    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding, n_results=top,
        include=["documents", "metadatas", "distances"],
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    hot_hits = [(d, m, 1 - dist) for d, m, dist in zip(docs, metas, dists) if (1 - dist) >= HOT_ACCEPT_THRESHOLD]
    if hot_hits:
        return "\n".join(f"[{m['scope']}/{m['type']}, sim={sim:.2f}] {d}" for d, m, sim in hot_hits)

    warm_hits = _warm_semantic_scan(query)
    if warm_hits:
        for d in warm_hits:
            _restore_from_cold(d)
        return "\n".join(f"[{d['scope']}/{d['type']}] {d['content']} (from warm/Turso)" for d in warm_hits)

    cold_hits = _cold_warmup_scan(query)
    if cold_hits:
        for d in cold_hits:
            _restore_from_cold(d)
        return "\n".join(f"[{d['scope']}/{d['type']}] {d['content']} (auto-restored from cold storage)" for d in cold_hits)

    return "No semantic matches found (checked hot, warm, and cold tiers)."


CURATOR_MODEL = "llama3.2:1b"

CURATION_PROMPT = """Task: {task}

Retrieved memory entries:
{candidates}

Which of these entries, if any, are actually USEFUL for answering the
task above -- not just about the same general topic, but something that
would actually help produce a correct answer.

Example: if the task asks about Python's try/except syntax, an entry
about "Python was created in 1991" mentions Python but is NOT useful --
it doesn't help answer the actual question. Do not keep an entry just
because it shares a keyword with the task.

Return ONLY the useful entries, copied exactly as given, one per line.
If none are useful, respond with exactly: NONE

Do not explain your reasoning. Do not add anything not in the original entries."""


CURATOR_AUTO_ACCEPT_THRESHOLD = 0.75  # real similarity score, skip the LLM entirely above this
CURATOR_FALLBACK_MODEL = "llama3.1:8b"  # cascade target when cheap curation fails verification


def _words(text: str) -> set:
    """Extracts real content words, ignoring the [scope/type, sim=X.XX]
    metadata tag -- found via testing that the tag alone can trivially
    satisfy a substring/overlap check, letting a curator response that
    dropped all real content still pass verification."""
    content_only = re.sub(r"^\[[^\]]*\]\s*", "", text.strip())
    return set(w.lower().strip(".,;:!?()[]{}\"'") for w in content_only.split() if len(w) > 3)


MIN_CANDIDATE_COVERAGE = 0.4  # output must retain this much of the matched candidate's real content


def _fuzzy_verify(curated: str, candidate_lines: list) -> list:
    """Only keep curator output lines that genuinely overlap a real
    candidate AND retain a meaningful share of its actual content.

    Checking overlap alone (what fraction of the output's words are
    real) isn't enough -- found via testing that a single-word output
    trivially scores a perfect overlap against any candidate containing
    that word, letting a curator response that shredded a full useful
    sentence down to one keyword pass verification as if nothing was
    lost. Coverage (what fraction of the CANDIDATE's real content
    survived in the output) catches exactly this: a full, accurately
    preserved sentence has high coverage; a single extracted keyword
    does not, even though both can have perfect overlap."""
    verified = []
    for out_line in curated.splitlines():
        out_line = out_line.strip()
        if not out_line or out_line.upper() == "NONE":
            continue
        out_words = _words(out_line)
        if not out_words:
            continue
        best_score = 0.0
        for cand in candidate_lines:
            cand_words = _words(cand)
            if not cand_words:
                continue
            shared = len(out_words & cand_words)
            overlap = shared / len(out_words)
            coverage = shared / len(cand_words)
            if overlap >= 0.6 and coverage >= MIN_CANDIDATE_COVERAGE:
                best_score = max(best_score, min(overlap, 1.0))
        if best_score > 0:
            verified.append(out_line)
    return verified


def _run_curator_pass(model: str, task: str, candidates_text: str, candidate_lines: list) -> list:
    """One curation attempt with a given model, verified against real
    candidates. Returns a list of verified-kept lines (possibly empty).

    Logs real token cost for every call, cheap or capable-fallback --
    curation calls were previously completely invisible in the metrics/
    dashboard, a real gap given how much emphasis the rest of this
    system puts on measured, not estimated, cost accounting."""
    prompt = CURATION_PROMPT.format(task=task, candidates=candidates_text)
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
            timeout=30,
        ).json()
        curated = resp.get("response", "").strip()
        input_tokens = resp.get("prompt_eval_count", 0) or 0
        output_tokens = resp.get("eval_count", 0) or 0
    except Exception:
        metrics.log_turn(
            task_snippet=task, category="_memory_curation", model=model,
            duration_seconds=0, error_occurred=True,
            assertion_flags="curation_call_failed",
            cascade_tier="curation_cheap" if model == CURATOR_MODEL else "curation_capable",
        )
        return None  # signals "curation call itself failed" -- different from "found nothing"

    verified = _fuzzy_verify(curated, candidate_lines) if curated else []
    metrics.log_turn(
        task_snippet=task, category="_memory_curation", model=model,
        duration_seconds=0, input_tokens=input_tokens, output_tokens=output_tokens,
        cascade_tier="curation_cheap" if model == CURATOR_MODEL else "curation_capable",
        escalated=(model != CURATOR_MODEL),
    )
    return verified


def curate_memory_context(task: str, raw_memory_text: str) -> str:
    """Filters retrieved memory down to what's actually relevant, in two
    layers:

    1. Deterministic: candidates carrying a real similarity score
       (from memory_semantic_search's hot-tier results, marked
       'sim=X.XX') above CURATOR_AUTO_ACCEPT_THRESHOLD are auto-kept --
       no LLM judgment call needed at all for clearly-relevant matches.
       This removes the exact failure class found in testing: a cheap
       model making a wrong keep/drop call on a single ambiguous
       candidate. High-confidence matches never reach that judgment call.

    2. LLM curation, only for genuinely ambiguous candidates (no score,
       or a score below the auto-accept bar) -- and even then, verified
       against the real candidate text rather than trusted, with a
       cascade fallback to a more capable model if the cheap curator's
       output doesn't survive verification (mirrors the main task
       cascade's own escalate-on-real-failure design)."""
    if not raw_memory_text or "No matching memories" in raw_memory_text or \
       "No semantic matches" in raw_memory_text:
        return raw_memory_text

    all_lines = [l.strip() for l in raw_memory_text.splitlines() if l.strip()]

    auto_kept = []
    needs_review = []
    for line in all_lines:
        m = re.search(r"sim=([0-9.]+)", line)
        if m and float(m.group(1)) >= CURATOR_AUTO_ACCEPT_THRESHOLD:
            auto_kept.append(line)
        else:
            needs_review.append(line)

    if not needs_review:
        return "\n".join(auto_kept)

    review_text = "\n".join(needs_review)
    cheap_result = _run_curator_pass(CURATOR_MODEL, task, review_text, needs_review)

    if cheap_result is None:
        # Curation call itself failed (network/timeout) -- fall back to
        # returning everything under review rather than silently losing it.
        return "\n".join(auto_kept + needs_review)

    if cheap_result:
        return "\n".join(auto_kept + cheap_result)

    # Cheap curator found nothing usable -- could be a correct "none of
    # these are relevant" call, or the cheap model failing on an
    # ambiguous case (the real failure mode found in testing). Escalate
    # to a more capable model for a second opinion rather than trusting
    # the cheap model's empty result outright.
    capable_result = _run_curator_pass(CURATOR_FALLBACK_MODEL, task, review_text, needs_review)
    if capable_result is None or not capable_result:
        return "\n".join(auto_kept)  # both agree, or capable also failed/confirmed nothing

    return "\n".join(auto_kept + capable_result)


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
