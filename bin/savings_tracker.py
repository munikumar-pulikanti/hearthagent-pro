#!/usr/bin/env python3
"""Track task evals and memory savings."""

import argparse
import json
import sqlite3
from pathlib import Path
from datetime import datetime


class SavingsTracker:
    """Log and track task quality + savings."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path.home() / ".ai-memory-vault" / "savings_tracker.db"
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize tracking database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS task_logs (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                task_type TEXT,
                hits INTEGER,
                source TEXT,
                quality TEXT,
                tool_calls INTEGER,
                description TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def log(
        self,
        task_type: str,
        hits: int,
        source: str,
        quality: str,
        tool_calls: int,
        description: str,
    ):
        """Log a task evaluation."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO task_logs
            (timestamp, task_type, hits, source, quality, tool_calls, description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(),
                task_type,
                hits,
                source,
                quality,
                tool_calls,
                description,
            ),
        )
        conn.commit()
        conn.close()
        print(f"Logged: {task_type} ({quality})")

    def report(self):
        """Generate savings report."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT task_type, quality, COUNT(*) as count FROM task_logs GROUP BY task_type, quality"
        ).fetchall()
        conn.close()

        print("\nSavings Report:")
        for task_type, quality, count in rows:
            print(f"  {task_type:15} {quality:12} {count:3}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Savings tracker")
    subparsers = parser.add_subparsers(dest="command")

    log_parser = subparsers.add_parser("log")
    log_parser.add_argument("--type", required=True)
    log_parser.add_argument("--hits", type=int, required=True)
    log_parser.add_argument("--source", required=True)
    log_parser.add_argument("--quality", required=True)
    log_parser.add_argument("--tool-calls", type=int, default=0)
    log_parser.add_argument("--desc", default="")

    subparsers.add_parser("report")

    args = parser.parse_args()
    tracker = SavingsTracker()

    if args.command == "log":
        tracker.log(
            args.type,
            args.hits,
            args.source,
            args.quality,
            args.tool_calls,
            args.desc,
        )
    elif args.command == "report":
        tracker.report()
    else:
        parser.print_help()
