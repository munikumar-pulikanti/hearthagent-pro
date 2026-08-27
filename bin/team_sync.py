"""Warm tier: push/pull memories between local SQLite and Turso (shared,
multi-writer). Credentials read from .env, never hardcoded."""
import os
import sqlite3
from pathlib import Path

import libsql_experimental as libsql
from dotenv import load_dotenv

load_dotenv()

VAULT_DB = Path.home() / ".ai-memory-vault" / "global_brain.db"
TURSO_URL = os.environ.get("TURSO_URL")
TURSO_AUTH_TOKEN = os.environ.get("TURSO_AUTH_TOKEN")

MEMORY_COLUMNS = [
    "id", "scope", "project", "type", "tags", "content",
    "created_at", "updated_at", "confidence", "archived",
]


def get_remote():
    if not TURSO_URL or not TURSO_AUTH_TOKEN:
        raise RuntimeError("TURSO_URL / TURSO_AUTH_TOKEN not set -- check your .env file")
    return libsql.connect("local-warm-replica.db", sync_url=TURSO_URL, auth_token=TURSO_AUTH_TOKEN)


def get_local():
    conn = sqlite3.connect(VAULT_DB)
    conn.row_factory = sqlite3.Row
    return conn


def push():
    """Send local memories that don't exist remotely (by id) up to Turso."""
    local = get_local()
    local_rows = local.execute("SELECT * FROM memories").fetchall()
    local.close()

    remote = get_remote()
    remote.sync()
    existing_ids = {r[0] for r in remote.execute("SELECT id FROM memories").fetchall()}

    pushed = 0
    for row in local_rows:
        if row["id"] in existing_ids:
            continue
        values = tuple(row[c] if c in row.keys() else None for c in MEMORY_COLUMNS)
        placeholders = ",".join("?" for _ in MEMORY_COLUMNS)
        remote.execute(
            f"INSERT INTO memories ({','.join(MEMORY_COLUMNS)}) VALUES ({placeholders})",
            values,
        )
        pushed += 1

    remote.commit()
    remote.sync()
    print(f"Pushed {pushed} new memories to Turso ({len(local_rows)} checked locally).")


def pull():
    """Bring remote memories that don't exist locally (by id) down to SQLite."""
    remote = get_remote()
    remote.sync()
    remote_rows = remote.execute("SELECT * FROM memories").fetchall()
    col_names = [d[0] for d in remote.execute("SELECT * FROM memories LIMIT 0").description] \
        if remote_rows else MEMORY_COLUMNS

    local = get_local()
    existing_ids = {r["id"] for r in local.execute("SELECT id FROM memories").fetchall()}

    pulled = 0
    for row in remote_rows:
        row_dict = dict(zip(MEMORY_COLUMNS, row))
        if row_dict["id"] in existing_ids:
            continue
        placeholders = ",".join("?" for _ in MEMORY_COLUMNS)
        local.execute(
            f"INSERT INTO memories ({','.join(MEMORY_COLUMNS)}) VALUES ({placeholders})",
            tuple(row_dict[c] for c in MEMORY_COLUMNS),
        )
        pulled += 1

    local.commit()
    local.close()
    print(f"Pulled {pulled} new memories from Turso ({len(remote_rows)} checked remotely).")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] not in ("push", "pull"):
        print("Usage: python3 bin/team_sync.py [push|pull]")
        sys.exit(1)
    if sys.argv[1] == "push":
        push()
    else:
        pull()
