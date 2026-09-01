"""Usage metrics for hearthagent-pro -- tracks routing decisions, model
usage, and which memory tier (hot/warm/cold/none) resolved each lookup."""
import sqlite3
import requests
from pathlib import Path

METRICS_DB = Path.home() / ".ai-memory-vault" / "hearthagent_pro_metrics.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY,
    timestamp INTEGER DEFAULT (unixepoch()),
    task_snippet TEXT,
    category TEXT,
    model TEXT,
    duration_seconds REAL,
    tool_call_count INTEGER DEFAULT 0,
    memory_pre_hit INTEGER DEFAULT 0,
    memory_tier TEXT DEFAULT 'none',
    error_occurred INTEGER DEFAULT 0,
    assertion_flags TEXT DEFAULT '',
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cascade_tier TEXT DEFAULT '',
    shortcut_fired INTEGER DEFAULT 0,
    model_digest TEXT DEFAULT '',
    escalated INTEGER DEFAULT 0,
    cheap_attempt_tokens INTEGER DEFAULT 0,
    capable_attempt_tokens INTEGER DEFAULT 0
);
"""


_model_digest_cache = {}


def get_model_digest(model_name: str) -> str:
    """Fetch a model's real digest from Ollama's /api/tags, cached per
    process. Logging this alongside the model name means a silent model
    update (same tag, different underlying weights) is at least visible
    in the data, even if nothing automatically reacts to it."""
    if model_name in _model_digest_cache:
        return _model_digest_cache[model_name]
    digest = ""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5).json()
        for m in resp.get("models", []):
            if m.get("name") == model_name or m.get("model") == model_name:
                digest = m.get("digest", "")[:12]  # short form is enough to detect a change
                break
    except Exception:
        pass
    _model_digest_cache[model_name] = digest
    return digest


def init_db():
    METRICS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(METRICS_DB)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def log_turn(task_snippet, category, model, duration_seconds, tool_call_count=0,
             memory_pre_hit=False, memory_tier="none", error_occurred=False,
             assertion_flags="", input_tokens=0, output_tokens=0,
             cascade_tier="", escalated=False, cheap_attempt_tokens=0,
             capable_attempt_tokens=0, shortcut_fired=False, digest_override=None):
    """digest_override lets a caller supply a fingerprint that covers
    more than just model weights (e.g. model + system prompt + tool
    schema combined) -- anything that changes the model's effective
    behavior should invalidate accumulated stats the same way a weight
    swap does, not just the weights alone."""
    init_db()
    model_digest = digest_override if digest_override is not None else get_model_digest(model)
    conn = sqlite3.connect(METRICS_DB)
    conn.execute(
        "INSERT INTO turns (task_snippet, category, model, duration_seconds, "
        "tool_call_count, memory_pre_hit, memory_tier, error_occurred, assertion_flags, "
        "input_tokens, output_tokens, cascade_tier, escalated, cheap_attempt_tokens, "
        "capable_attempt_tokens, shortcut_fired, model_digest) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (task_snippet[:200], category, model, duration_seconds, tool_call_count,
         int(memory_pre_hit), memory_tier, int(error_occurred), assertion_flags,
         input_tokens, output_tokens, cascade_tier, int(escalated),
         cheap_attempt_tokens, capable_attempt_tokens, int(shortcut_fired), model_digest)
    )
    conn.commit()
    conn.close()


def all_turns():
    init_db()
    conn = sqlite3.connect(METRICS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM turns ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def category_breakdown():
    turns = all_turns()
    counts = {}
    for t in turns:
        counts[t["category"]] = counts.get(t["category"], 0) + 1
    return counts


def model_breakdown():
    turns = all_turns()
    counts = {}
    for t in turns:
        counts[t["model"]] = counts.get(t["model"], 0) + 1
    return counts


def memory_tier_breakdown():
    turns = all_turns()
    counts = {"hot": 0, "warm": 0, "cold": 0, "none": 0}
    for t in turns:
        tier = t["memory_tier"] if t["memory_tier"] in counts else "none"
        counts[tier] += 1
    return counts


# ---------------- Eval baselines (confirmed-good outcomes) ----------------

EVAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_baselines (
    id INTEGER PRIMARY KEY,
    timestamp INTEGER DEFAULT (unixepoch()),
    task TEXT,
    expected_summary TEXT,
    git_commit TEXT
);

CREATE TABLE IF NOT EXISTS eval_results (
    id INTEGER PRIMARY KEY,
    timestamp INTEGER DEFAULT (unixepoch()),
    baseline_id INTEGER,
    passed INTEGER,
    judge_reasoning TEXT,
    actual_summary TEXT
);
"""


def init_eval_db():
    init_db()
    conn = sqlite3.connect(METRICS_DB)
    conn.executescript(EVAL_SCHEMA)
    conn.commit()
    conn.close()


def save_eval_baseline(task: str, expected_summary: str, git_commit: str = "") -> int:
    init_eval_db()
    conn = sqlite3.connect(METRICS_DB)
    cursor = conn.execute(
        "INSERT INTO eval_baselines (task, expected_summary, git_commit) VALUES (?, ?, ?)",
        (task, expected_summary, git_commit)
    )
    baseline_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return baseline_id


def all_baselines():
    init_eval_db()
    conn = sqlite3.connect(METRICS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM eval_baselines ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_eval_result(baseline_id: int, passed: bool, judge_reasoning: str, actual_summary: str):
    init_eval_db()
    conn = sqlite3.connect(METRICS_DB)
    conn.execute(
        "INSERT INTO eval_results (baseline_id, passed, judge_reasoning, actual_summary) "
        "VALUES (?, ?, ?, ?)",
        (baseline_id, int(passed), judge_reasoning, actual_summary)
    )
    conn.commit()
    conn.close()


def all_eval_results():
    init_eval_db()
    conn = sqlite3.connect(METRICS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT r.*, b.task FROM eval_results r "
        "JOIN eval_baselines b ON b.id = r.baseline_id "
        "ORDER BY r.id DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


ESCALATION_RATE_WINDOW = 50  # most recent turns per category, not all-time


def category_escalation_rate(category: str, model: str = None, digest_override: str = None) -> dict:
    """Rolling recent-window escalation rate for a category -- deliberately
    NOT an all-time average. An unweighted all-time average has a real flaw:
    as history accumulates, fresh evidence from the shortcut's own holdout
    traffic gets diluted into irrelevance by a growing pile of old data,
    even though the holdout keeps running. Using only the most recent
    ESCALATION_RATE_WINDOW turns means new evidence always has real,
    consistent influence on the number, not shrinking influence over time.

    Also filters by the CURRENT digest, when one is available. Found via
    real review: logging model_digest gave visibility into a silent
    model swap, but nothing actually USED it. Found via a real follow-up
    review: filtering by model weights alone is still incomplete -- a
    system prompt edit or a tool being added/changed/removed shifts the
    win rate just as much as a weight change does. digest_override lets
    a caller supply a wider fingerprint (model + prompt + tool schema
    combined) instead of just the bare model-weights digest -- pass
    digest_override when available; it takes priority over `model`."""
    turns = all_turns()  # already ordered by id DESC (most recent first)
    if digest_override is not None:
        current_digest = digest_override
    elif model:
        current_digest = get_model_digest(model)
    else:
        current_digest = None

    # Only count turns where the cheap tier was actually attempted.
    # Shortcut-skipped turns (cheap_attempt_tokens is None) are always
    # logged as escalated=True by construction, not because cheap tier
    # was tried and failed. Including them would make the rate
    # self-reinforcing: shortcut fires -> more fake-escalated rows ->
    # rate stays pinned high -> shortcut keeps firing, even if the
    # underlying model improved and cheap tier would now succeed.
    category_turns = [
        t for t in turns
        if t["category"] == category
        and t["cheap_attempt_tokens"] is not None
        and (current_digest is None or t.get("model_digest") == current_digest)
    ][:ESCALATION_RATE_WINDOW]
    if not category_turns:
        return {"sample_size": 0, "escalation_rate": None, "escalation_rate_lower_bound": None}
    escalated_count = sum(1 for t in category_turns if t["escalated"])
    n = len(category_turns)
    raw_rate = escalated_count / n
    lower_bound = _wilson_lower_bound(escalated_count, n)
    return {
        "sample_size": n,
        "escalation_rate": raw_rate,
        "escalation_rate_lower_bound": lower_bound,
    }


def _wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    """95% Wilson score interval lower bound for a proportion. Using this
    instead of the raw fraction means small samples can't thrash a
    threshold-based decision purely from chance: a true underlying rate
    of, say, 65% can easily show 80%+ in a sample of 20 by luck alone.
    The lower bound asks 'am I actually confident the true rate clears
    the threshold', not just 'did this one sample clear it'."""
    if n == 0:
        return 0.0
    p_hat = successes / n
    denom = 1 + z * z / n
    center = p_hat + z * z / (2 * n)
    margin = z * ((p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, (center - margin) / denom)
