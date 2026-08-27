"""Tool definitions: file I/O, shell execution, memory queries."""

from pathlib import Path
import subprocess
from typing import Any, Dict


def read_file(path: str) -> str:
    """Read file contents."""
    try:
        return Path(path).read_text()
    except Exception as e:
        return f"Error reading {path}: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to file."""
    try:
        Path(path).write_text(content)
        return f"Wrote to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


def list_dir(path: str) -> list:
    """List directory contents."""
    try:
        return [str(p) for p in Path(path).iterdir()]
    except Exception as e:
        return [f"Error listing {path}: {e}"]


def run_shell(command: str) -> str:
    """Execute shell command."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout or result.stderr
    except subprocess.TimeoutExpired:
        return f"Command timed out: {command}"
    except Exception as e:
        return f"Error running {command}: {e}"


def search_code(query: str, directory: str = ".") -> list:
    """Search code for query string."""
    # TODO: Implement semantic code search or grep
    return []


TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "run_shell": run_shell,
    "search_code": search_code,
}
