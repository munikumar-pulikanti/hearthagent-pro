"""LangGraph ReAct agent for hearthagent-pro. Routes each task through
ModelRouter, checks memory deterministically before any model call, and
logs usage metrics -- including which memory tier resolved the lookup."""
import json
import os
import re
import time
import random
import random

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


FIDELITY_CHECKED_TOOLS = {"list_dir_tool", "search_code_tool", "memory_search_tool", "web_search_tool"}


def check_tool_result_fidelity(final_content: str, messages: list) -> list:
    """For tools that return a concrete, enumerable list (files, search
    hits, memory entries), verify the model's narrative summary doesn't
    mention items that don't actually appear in the real tool output.
    Found via real testing: llama3.2:1b correctly executed list_dir_tool
    but then fabricated two filenames in its own summary of the real
    result. Deterministic, no extra LLM call."""
    flags = []
    if not final_content:
        return flags

    for msg in messages:
        tool_name = getattr(msg, "name", None)
        if tool_name not in FIDELITY_CHECKED_TOOLS:
            continue
        tool_content = getattr(msg, "content", "") or ""
        if not tool_content or "No matches found" in tool_content or "No results found" in tool_content:
            continue

        real_lines = set(
            l.strip().lstrip("fd").strip()
            for l in tool_content.splitlines() if l.strip()
        )
        mentioned = set(re.findall(r'\b[\w][\w.-]*\.\w{1,6}\b', final_content))
        fabricated = [m for m in mentioned if m not in real_lines and not any(m in rl for rl in real_lines)]

        if fabricated:
            flags.append(f"fabricated_items_in_summary:{','.join(sorted(fabricated)[:3])}")

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


CASCADE_CHEAP_MODEL = "llama3.2:1b"
CASCADE_CAPABLE_MODEL = "llama3.1:8b"
SHORTCUT_MIN_SAMPLES = 20
SHORTCUT_ESCALATION_THRESHOLD = 0.8
SHORTCUT_SAMPLE_RATE = 0.2


def _attempt(model_name: str, category: str, messages: list) -> dict:
    """Run one cascade attempt with a given model. Returns everything
    needed to decide whether to escalate and what to log, without
    mutating the caller's history -- the caller decides what to keep."""
    agent = build_agent(model_name, category)

    try:
        result = agent.invoke({"messages": messages}, config={"recursion_limit": 15})
    except GraphRecursionError:
        return {"error": "recursion_limit", "messages": messages}
    except Exception as e:
        return {"error": str(e), "messages": messages}

    final_msg = result["messages"][-1]
    tool_call_count = sum(len(getattr(m, "tool_calls", None) or []) for m in result["messages"])
    tool_names_called = {
        tc.get("name") for m in result["messages"]
        for tc in (getattr(m, "tool_calls", None) or [])
    }
    input_tokens = sum((getattr(m, "usage_metadata", None) or {}).get("input_tokens", 0) or 0 for m in result["messages"])
    output_tokens = sum((getattr(m, "usage_metadata", None) or {}).get("output_tokens", 0) or 0 for m in result["messages"])

    tool_name, args = try_parse_manual_tool_call(final_msg.content or "")
    if tool_name:
        tool_result = TOOLS_BY_NAME[tool_name].invoke(args)
        followup = result["messages"] + [
            ("assistant", f"[executed {tool_name} manually after it was printed as text instead of called]"),
            ("tool", str(tool_result)),
        ]
        try:
            result2 = agent.invoke({"messages": followup}, config={"recursion_limit": 15})
        except Exception:
            result2 = result
        input_tokens += sum((getattr(m, "usage_metadata", None) or {}).get("input_tokens", 0) or 0 for m in result2["messages"])
        output_tokens += sum((getattr(m, "usage_metadata", None) or {}).get("output_tokens", 0) or 0 for m in result2["messages"])
        final_content = result2["messages"][-1].content
        out_messages = result2["messages"]
    else:
        final_content = final_msg.content
        out_messages = result["messages"]

    return {
        "error": None,
        "final_content": final_content,
        "messages": out_messages,
        "tool_call_count": tool_call_count,
        "tool_names_called": tool_names_called,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


class Session:
    def __init__(self):
        self.router = ModelRouter()
        self.history = []

    def send(self, user_input: str) -> str:
        start = time.time()
        _, category = self.router.route(user_input)

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

        if memory_pre_hit:
            print(f"--- memory pre-check hit ({memory_tier}), injecting before model call ---")
            augmented_input = (
                f"{user_input}\n\n"
                f"[Relevant memory found before you were called -- use this if helpful, "
                f"don't re-search unless it's insufficient:]\n{pre_hit}"
            )
        else:
            augmented_input = user_input

        base_messages = self.history + [("user", augmented_input)]

        stats = metrics.category_escalation_rate(category)
        shortcut_eligible = (
            stats["sample_size"] >= SHORTCUT_MIN_SAMPLES
            and stats["escalation_rate"] is not None
            and stats["escalation_rate"] >= SHORTCUT_ESCALATION_THRESHOLD
        )
        skip_cheap_this_time = shortcut_eligible and random.random() > SHORTCUT_SAMPLE_RATE

        if skip_cheap_this_time:
            print(f"--- shortcut: category '{category}' escalates {stats['escalation_rate']:.0%} of the time "
                  f"(n={stats['sample_size']}), skipping cheap tier this turn ---")
            capable = _attempt(CASCADE_CAPABLE_MODEL, category, base_messages)
            if capable["error"] is not None:
                duration = time.time() - start
                metrics.log_turn(
                    task_snippet=user_input, category=category, model=CASCADE_CAPABLE_MODEL,
                    duration_seconds=duration, memory_pre_hit=memory_pre_hit,
                    memory_tier=memory_tier, error_occurred=True,
                    assertion_flags="capable_tier_error:" + capable["error"],
                    cascade_tier="capable", escalated=True,
                    cheap_attempt_tokens=None, capable_attempt_tokens=0,
                )
                if capable["error"] == "recursion_limit":
                    return "I got stuck in a loop after too many steps trying to answer that. Try rephrasing, or breaking it into a smaller task."
                raise RuntimeError(capable["error"])

            capable_flags = (
                check_assertions(capable["final_content"], capable["tool_call_count"], capable["tool_names_called"])
                + check_faithfulness(capable["final_content"], memory_pre_hit, pre_hit)
                + check_tool_result_fidelity(capable["final_content"], capable["messages"])
            )
            if capable_flags:
                print(f"--- ASSERTION FAILURES (capable tier, shortcut path): {', '.join(capable_flags)} ---")

            for m in capable["messages"][len(base_messages):]:
                m.pretty_print()

            self.history = capable["messages"]
            capable_tokens = capable["input_tokens"] + capable["output_tokens"]
            duration = time.time() - start
            metrics.log_turn(
                task_snippet=user_input, category=category, model=CASCADE_CAPABLE_MODEL,
                duration_seconds=duration, tool_call_count=capable["tool_call_count"],
                memory_pre_hit=memory_pre_hit, memory_tier=memory_tier,
                error_occurred=False, assertion_flags="shortcut_skip:" + ",".join(capable_flags) if capable_flags else "shortcut_skip",
                input_tokens=capable["input_tokens"], output_tokens=capable["output_tokens"],
                cascade_tier="capable", escalated=True,
                cheap_attempt_tokens=None, capable_attempt_tokens=capable_tokens,
            )
            return capable["final_content"]

        print(f"--- cascade: trying cheap tier ({CASCADE_CHEAP_MODEL}) ---")
        cheap = _attempt(CASCADE_CHEAP_MODEL, category, base_messages)

        cheap_tokens = 0
        if cheap["error"] is None:
            cheap_tokens = cheap["input_tokens"] + cheap["output_tokens"]
            cheap_flags = (
                check_assertions(cheap["final_content"], cheap["tool_call_count"], cheap["tool_names_called"])
                + check_faithfulness(cheap["final_content"], memory_pre_hit, pre_hit)
                + check_tool_result_fidelity(cheap["final_content"], cheap["messages"])
            )
        else:
            cheap_flags = [f"cheap_tier_error:{cheap['error']}"]

        escalated = bool(cheap["error"]) or bool(cheap_flags)

        if not escalated:
            print(f"--- cascade: cheap tier passed checks, no escalation ---")
            for m in cheap["messages"][len(base_messages):]:
                m.pretty_print()
            self.history = cheap["messages"]
            final_content = cheap["final_content"]
            duration = time.time() - start
            metrics.log_turn(
                task_snippet=user_input, category=category, model=CASCADE_CHEAP_MODEL,
                duration_seconds=duration, tool_call_count=cheap["tool_call_count"],
                memory_pre_hit=memory_pre_hit, memory_tier=memory_tier,
                error_occurred=False, assertion_flags="",
                input_tokens=cheap["input_tokens"], output_tokens=cheap["output_tokens"],
                cascade_tier="cheap", escalated=False,
                cheap_attempt_tokens=cheap_tokens, capable_attempt_tokens=0,
            )
            return final_content

        print(f"--- cascade: cheap tier failed checks ({', '.join(cheap_flags)}), escalating to {CASCADE_CAPABLE_MODEL} ---")
        capable = _attempt(CASCADE_CAPABLE_MODEL, category, base_messages)

        if capable["error"] is not None:
            duration = time.time() - start
            metrics.log_turn(
                task_snippet=user_input, category=category, model=CASCADE_CAPABLE_MODEL,
                duration_seconds=duration, memory_pre_hit=memory_pre_hit,
                memory_tier=memory_tier, error_occurred=True,
                assertion_flags="capable_tier_error:" + capable["error"],
                cascade_tier="capable", escalated=True,
                cheap_attempt_tokens=cheap_tokens, capable_attempt_tokens=0,
            )
            if capable["error"] == "recursion_limit":
                return "I got stuck in a loop after too many steps trying to answer that. Try rephrasing, or breaking it into a smaller task."
            raise RuntimeError(capable["error"])

        capable_flags = (
            check_assertions(capable["final_content"], capable["tool_call_count"], capable["tool_names_called"])
            + check_faithfulness(capable["final_content"], memory_pre_hit, pre_hit)
            + check_tool_result_fidelity(capable["final_content"], capable["messages"])
        )
        if capable_flags:
            print(f"--- ASSERTION FAILURES (capable tier): {', '.join(capable_flags)} ---")

        for m in capable["messages"][len(base_messages):]:
            m.pretty_print()

        self.history = capable["messages"]
        final_content = capable["final_content"]
        capable_tokens = capable["input_tokens"] + capable["output_tokens"]
        duration = time.time() - start

        metrics.log_turn(
            task_snippet=user_input, category=category, model=CASCADE_CAPABLE_MODEL,
            duration_seconds=duration, tool_call_count=capable["tool_call_count"],
            memory_pre_hit=memory_pre_hit, memory_tier=memory_tier,
            error_occurred=False, assertion_flags="escalated_due_to:" + ",".join(cheap_flags) + (";capable_flags:" + ",".join(capable_flags) if capable_flags else ""),
            input_tokens=capable["input_tokens"], output_tokens=capable["output_tokens"],
            cascade_tier="capable", escalated=True,
            cheap_attempt_tokens=cheap_tokens, capable_attempt_tokens=capable_tokens,
        )
        return final_content


def run_task(task: str):
    session = Session()
    session.send(task)
