"""Usage metrics for hearthagent-pro -- tracks routing decisions, model
usage, and which memory tier (hot/warm/cold/none) resolved each lookup."""
import sqlite3
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
    escalated INTEGER DEFAULT 0,
    cheap_attempt_tokens INTEGER DEFAULT 0,
    capable_attempt_tokens INTEGER DEFAULT 0
);
"""


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
             capable_attempt_tokens=0):
    init_db()
    conn = sqlite3.connect(METRICS_DB)
    conn.execute(
        "INSERT INTO turns (task_snippet, category, model, duration_seconds, "
        "tool_call_count, memory_pre_hit, memory_tier, error_occurred, assertion_flags, "
        "input_tokens, output_tokens, cascade_tier, escalated, cheap_attempt_tokens, "
        "capable_attempt_tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (task_snippet[:200], category, model, duration_seconds, tool_call_count,
         int(memory_pre_hit), memory_tier, int(error_occurred), assertion_flags,
         input_tokens, output_tokens, cascade_tier, int(escalated),
         cheap_attempt_tokens, capable_attempt_tokens)
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


def category_escalation_rate(category: str) -> dict:
    """Rolling recent-window escalation rate for a category -- deliberately
    NOT an all-time average. An unweighted all-time average has a real flaw:
    as history accumulates, fresh evidence from the shortcut's own holdout
    traffic gets diluted into irrelevance by a growing pile of old data,
    even though the holdout keeps running. Using only the most recent
    ESCALATION_RATE_WINDOW turns means new evidence always has real,
    consistent influence on the number, not shrinking influence over time."""
    turns = all_turns()  # already ordered by id DESC (most recent first)
    category_turns = [t for t in turns if t["category"] == category][:ESCALATION_RATE_WINDOW]
    if not category_turns:
        return {"sample_size": 0, "escalation_rate": None}
    escalated_count = sum(1 for t in category_turns if t["escalated"])
    return {
        "sample_size": len(category_turns),
        "escalation_rate": escalated_count / len(category_turns),
    }
