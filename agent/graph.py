"""LangGraph ReAct agent for hearthagent-pro. Routes each task through
ModelRouter, checks memory deterministically before any model call, and
logs usage metrics -- including which memory tier resolved the lookup."""
import json
import os
import re
import time

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError

from agent.tools import (
    read_file, write_file, list_dir, search_code,
    run_shell, memory_search, memory_save,
    memory_sync_embeddings, memory_semantic_search,
    web_search, curate_memory_context,
)
from agent.router import ModelRouter
from agent import metrics
CURATOR_MODEL_LABEL = "llama3.2:1b curator"


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
    """Meaning-based (semantic) search across hot, warm, and cold memory tiers."""
    return memory_semantic_search(query)


@tool
def memory_sync_embeddings_tool() -> str:
    """Re-index all active memory vault rows into ChromaDB."""
    return memory_sync_embeddings()


@tool
def memory_save_tool(scope: str, type_: str, content: str, tags: str = "", evidence_url: str = "") -> str:
    """Save a new finding to the persistent memory vault. If you have a
    real source for this finding (a file path, a command output, a URL),
    pass it as evidence_url. Findings without verified evidence can never
    be promoted past 'suspected' confidence, no matter how often they're
    restated -- only provide a URL if it's real and you actually checked it,
    never fabricate one just to raise confidence."""
    return memory_save(scope, type_, content, tags, evidence_url)


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


FABRICATED_SUCCESS_PHRASES = [
    "test passed", "tests pass", "successfully ran", "completed successfully",
    "ran successfully", "all tests passed",
]


def check_faithfulness(final_content: str, memory_pre_hit: bool, pre_hit_text: str) -> list:
    """If memory was injected but the answer ignores it entirely, that's
    a groundedness failure -- the model fabricated instead of using real
    context. Heuristic word-overlap check, deterministic, no extra LLM call."""
    if not memory_pre_hit or not pre_hit_text:
        return []
    memory_words = set(w.lower() for w in pre_hit_text.split() if len(w) > 4)
    answer_words = set(w.lower() for w in final_content.split() if len(w) > 4)
    overlap = memory_words & answer_words
    if len(overlap) < 2:
        return ["memory_injected_but_likely_ignored"]
    return []


def check_assertions(final_content: str, tool_call_count: int, tool_names_called: set) -> list:
    """Deterministic, local, no-cloud checks against a turn's output.
    Catches known bad patterns (like the qwen2.5-coder fabrication bug)
    without needing any external service."""
    flags = []
    lowered = (final_content or "").lower()

    claims_success = any(p in lowered for p in FABRICATED_SUCCESS_PHRASES)
    actually_verified = "run_shell_tool" in tool_names_called
    if claims_success and not actually_verified:
        # Writing "test passed" to a file, or just calling write_file_tool,
        # is NOT verification. Only actually running the test counts.
        flags.append("claimed_success_without_real_verification")

    if not final_content or not final_content.strip():
        flags.append("empty_response")

    if '"name":' in (final_content or "") and '"arguments":' in (final_content or ""):
        flags.append("unexecuted_tool_call_leaked_to_user")

    return flags


def _detect_memory_tier(pre_hit_text: str) -> str:
    """Infer which tier actually resolved a memory pre-check, from the
    markers memory_semantic_search already includes in its output."""
    if "auto-restored from cold storage" in pre_hit_text:
        return "cold"
    if "from warm/Turso" in pre_hit_text:
        return "warm"
    if pre_hit_text and pre_hit_text.startswith("ERROR"):
        return "none"
    if pre_hit_text and pre_hit_text.startswith("ERROR"):
        return "none"
    if pre_hit_text and "No matching memories" not in pre_hit_text and \
       "No semantic matches" not in pre_hit_text:
        return "hot"
    return "none"


class Session:
    def __init__(self):
        self.router = ModelRouter()
        self.history = []

    def send(self, user_input: str) -> str:
        start = time.time()
        model_name, category = self.router.route(user_input)
        print(f"--- routed to: {model_name} (category: {category}) ---")

        pre_hit = memory_search(user_input)
        if "No matching memories" in pre_hit:
            pre_hit = memory_semantic_search(user_input)
        memory_tier = _detect_memory_tier(pre_hit)

        if memory_tier != "none":
            curated = curate_memory_context(user_input, pre_hit)
            if curated:
                print(f"--- memory curated by {CURATOR_MODEL_LABEL}: kept relevant entries only ---")
                pre_hit = curated
            else:
                print(f"--- memory curated by {CURATOR_MODEL_LABEL}: nothing relevant, dropping retrieval ---")
                memory_tier = "none"

        memory_pre_hit = memory_tier != "none"

        agent = build_agent(model_name, category)

        if memory_pre_hit:
            print(f"--- memory pre-check hit ({memory_tier}), injecting before model call ---")
            augmented_input = (
                f"{user_input}\n\n"
                f"[Relevant memory found before you were called -- use this if helpful, "
                f"don't re-search unless it's insufficient:]\n{pre_hit}"
            )
            history_len_before = len(self.history)
            self.history.append(("user", augmented_input))
        else:
            history_len_before = len(self.history)
            self.history.append(("user", user_input))

        error_occurred = False
        try:
            result = agent.invoke({"messages": self.history}, config={"recursion_limit": 15})
        except GraphRecursionError:
            error_occurred = True
            duration = time.time() - start
            metrics.log_turn(
                task_snippet=user_input, category=category, model=model_name,
                duration_seconds=duration, memory_pre_hit=memory_pre_hit,
                memory_tier=memory_tier, error_occurred=True,
                assertion_flags="hit_recursion_limit",
            )
            return (f"I got stuck in a loop after too many steps trying to answer "
                    f"that (limit: 15). Try rephrasing, or breaking it into a "
                    f"smaller task.")
        except Exception:
            error_occurred = True
            duration = time.time() - start
            metrics.log_turn(
                task_snippet=user_input, category=category, model=model_name,
                duration_seconds=duration, memory_pre_hit=memory_pre_hit,
                memory_tier=memory_tier, error_occurred=True,
            )
            raise

        for m in result["messages"][history_len_before:]:
            m.pretty_print()

        final_msg = result["messages"][-1]
        tool_call_count = sum(
            len(getattr(m, "tool_calls", None) or []) for m in result["messages"]
        )
        tool_names_called = {tc.get("name") for m in result["messages"] for tc in (getattr(m, "tool_calls", None) or [])}
        input_tokens = sum((getattr(m, "usage_metadata", None) or {}).get("input_tokens", 0) or 0 for m in result["messages"])
        output_tokens = sum((getattr(m, "usage_metadata", None) or {}).get("output_tokens", 0) or 0 for m in result["messages"])
        tool_name, args = try_parse_manual_tool_call(final_msg.content or "")

        if tool_name:
            print(f"--- manual fallback: model printed a tool call as text, executing {tool_name}({args}) ---")
            tool_result = TOOLS_BY_NAME[tool_name].invoke(args)
            self.history = result["messages"] + [
                ("assistant", f"[executed {tool_name} manually after it was printed as text instead of called]"),
                ("tool", str(tool_result)),
            ]
            result2 = agent.invoke({"messages": self.history}, config={"recursion_limit": 15})
            for m in result2["messages"][len(result["messages"]):]:
                m.pretty_print()
            self.history = result2["messages"]
            final_content = result2["messages"][-1].content
        else:
            self.history = result["messages"]
            final_content = final_msg.content

        duration = time.time() - start
        faithfulness_flags = check_faithfulness(final_content, memory_pre_hit, pre_hit)
        assertion_flags = check_assertions(final_content, tool_call_count, tool_names_called)
        all_flags = assertion_flags + faithfulness_flags
        if all_flags:
            print(f"--- ASSERTION FAILURES: {', '.join(all_flags)} ---")

        metrics.log_turn(
            task_snippet=user_input, category=category, model=model_name,
            duration_seconds=duration, tool_call_count=tool_call_count,
            memory_pre_hit=memory_pre_hit, memory_tier=memory_tier,
            error_occurred=error_occurred, assertion_flags=",".join(all_flags),
            input_tokens=input_tokens, output_tokens=output_tokens,
        )

        return final_content


def run_task(task: str):
    session = Session()
    session.send(task)
