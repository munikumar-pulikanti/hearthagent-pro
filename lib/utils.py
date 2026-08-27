"""Utilities and helpers."""

import json
from datetime import datetime
from typing import Any, Dict


def to_json(obj: Any) -> str:
    """Serialize to JSON."""
    return json.dumps(obj, default=str)


def from_json(s: str) -> Any:
    """Deserialize from JSON."""
    return json.loads(s)


def timestamp() -> str:
    """Current ISO timestamp."""
    return datetime.now().isoformat()


def log_event(event_type: str, data: Dict[str, Any] = None):
    """Log event to console."""
    ts = timestamp()
    msg = f"[{ts}] {event_type}"
    if data:
        msg += f" {to_json(data)}"
    print(msg)
