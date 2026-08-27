"""LangGraph ReAct agent for hearthagent-pro. Routes each task through
ModelRouter, with a manual fallback for models that print a tool call as
text instead of actually invoking it (a documented issue with some
qwen2.5-coder sizes in this Ollama setup)."""
import json
import os
import re

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from agent.tools import (
    read_file, write_file, list_dir, search_code,
    run_shell, memory_search, memory_save,
    memory_sync_embeddings, memory_semantic_search,
    web_search,
)
from agent.router import ModelRouter


@tool
def read_file_tool(path: str) -> str:
    """Read and return the contents of a file at the given path."""
    return read_file(path)


@tool
def write_file_tool(path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed."""
    return write_file(path, content)


@tool
def list_dir_tool(path: str = ".") -> str:
    """List files and subdirectories at the given path."""
    return list_dir(path)


@tool
def search_code_tool(query: str, path: str = ".") -> str:
    """Search project files for a string or pattern."""
    return search_code(query, path)


@tool
def run_shell_tool(command: str) -> str:
    """Run an allowlisted shell command (ls, cat, pwd, grep, find, python, pytest, git, uv)."""
    return run_shell(command)


@tool
def memory_search_tool(query: str) -> str:
    """Fast keyword (FTS) search over the persistent memory vault. Use this first."""
    return memory_search(query)


@tool
def memory_semantic_search_tool(query: str) -> str:
    """Meaning-based (semantic) search over the memory vault via ChromaDB."""
    return memory_semantic_search(query)


@tool
def memory_sync_embeddings_tool() -> str:
    """Re-index all memory vault rows into ChromaDB. Run after saving new memories."""
    return memory_sync_embeddings()


@tool
def memory_save_tool(scope: str, type_: str, content: str, tags: str = "") -> str:
    """Save a new confirmed finding to the persistent memory vault."""
    return memory_save(scope, type_, content, tags)


@tool
def web_search_tool(query: str) -> str:
    """Search the live web for current information not in local files or memory."""
    return web_search(query)


TOOLS = [
    read_file_tool, write_file_tool, list_dir_tool,
    search_code_tool, run_shell_tool,
    memory_search_tool, memory_semantic_search_tool,
    memory_sync_embeddings_tool, memory_save_tool,
    web_search_tool,
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def try_parse_manual_tool_call(text):
    """Detect a tool call printed as JSON text instead of really being
    invoked. Returns (tool_name, args_dict) or (None, None)."""
    if not text:
        return None, None
    match = re.search(
        r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}',
        text,
    )
    if not match:
        return None, None
    name = match.group(1)
    if name not in TOOLS_BY_NAME:
        return None, None
    try:
        args = json.loads(match.group(2))
    except Exception:
        return None, None
    return name, args


def build_system_prompt(category: str) -> str:
    cwd = os.getcwd()
    return (
        f"You are hearthagent-pro, a local coding agent. Task category: {category}. "
        f"Your current working directory is: {cwd}. "
        f"When a task refers to 'this project' or 'here' with no explicit path, "
        f"use the path '.' -- never guess or invent a path. "
        f"You have tools to read/write files, list directories, search code, run a "
        f"small set of allowlisted shell commands, search/save to a persistent "
        f"memory vault, and search the live web. "
        f"For memory: try memory_search_tool first, then memory_semantic_search_tool "
        f"if that finds nothing. Only use web_search_tool as a last resort. "
        f"Be concise. Use tools rather than guessing at file contents. "
        f"Never fabricate a result -- if you need to verify something (like a test "
        f"passing), actually run it via run_shell_tool rather than claiming success."
    )


def build_agent(model_name: str, category: str):
    llm = ChatOllama(model=model_name, temperature=0)
    return create_react_agent(llm, TOOLS, prompt=build_system_prompt(category))


class Session:
    def __init__(self):
        self.router = ModelRouter()
        self.history = []

    def send(self, user_input: str) -> str:
        model_name, category = self.router.route(user_input)
        print(f"--- routed to: {model_name} (category: {category}) ---")

        agent = build_agent(model_name, category)
        self.history.append(("user", user_input))
        result = agent.invoke({"messages": self.history})

        for m in result["messages"]:
            m.pretty_print()

        final_msg = result["messages"][-1]
        tool_name, args = try_parse_manual_tool_call(final_msg.content or "")

        if tool_name:
            print(f"--- manual fallback: model printed a tool call as text, executing {tool_name}({args}) ---")
            tool_result = TOOLS_BY_NAME[tool_name].invoke(args)
            self.history = result["messages"] + [
                ("assistant", f"[executed {tool_name} manually after it was printed as text instead of called]"),
                ("tool", str(tool_result)),
            ]
            result2 = agent.invoke({"messages": self.history})
            for m in result2["messages"]:
                m.pretty_print()
            self.history = result2["messages"]
            return result2["messages"][-1].content

        self.history = result["messages"]
        return final_msg.content


def run_task(task: str):
    session = Session()
    session.send(task)
