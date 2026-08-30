"""MCP server exposing hearthagent-pro's memory layer to ANY MCP-compatible
tool (Antigravity, Windsurf, Cursor, Claude Desktop, Kiro, etc). All of
them hit the same SQLite/ChromaDB/Turso files -- memory is genuinely
shared across whichever tool you're using, without needing hearthagent-pro
itself to be running that session.

Run with: uv run python3 mcp_server.py
Then point any MCP client's config at this command.
"""
from mcp.server.fastmcp import FastMCP

from agent.tools import (
    memory_search, memory_save, memory_semantic_search,
    memory_sync_embeddings,
)

mcp = FastMCP("hearthagent-pro-memory")


@mcp.tool()
def search_memory(query: str) -> str:
    """Fast keyword search over the persistent, cross-tool memory vault."""
    return memory_search(query)


@mcp.tool()
def search_memory_semantic(query: str) -> str:
    """Meaning-based search over the memory vault, cascading hot/warm/cold tiers."""
    return memory_semantic_search(query)


@mcp.tool()
def save_memory(scope: str, type_: str, content: str, tags: str = "", evidence_url: str = "") -> str:
    """Save a confirmed finding to the shared memory vault. Provide
    evidence_url (a real, checkable source) if you have one -- findings
    without verified evidence cap at 'suspected' confidence."""
    return memory_save(scope, type_, content, tags, evidence_url)


@mcp.tool()
def sync_embeddings() -> str:
    """Re-index active memory rows into ChromaDB."""
    return memory_sync_embeddings()


if __name__ == "__main__":
    mcp.run()