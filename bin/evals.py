"""Eval runner: re-executes confirmed-good baseline tasks fresh, and uses
an LLM judge to check whether current agent behavior still matches.
Run with: uv run python3 bin/evals.py
"""
import requests

from agent.graph import Session
from agent import metrics

JUDGE_MODEL = "llama3.1:8b"

JUDGE_PROMPT = """You are grading whether a new answer matches the quality \
and correctness of a previously confirmed-good answer to the same task.

Task: {task}

Previously confirmed-good answer:
{expected}

New answer to grade:
{actual}

Does the new answer meet the same bar -- correct, complete, no fabrication? \
Reply in exactly this format:
VERDICT: PASS or FAIL
REASON: one sentence why
"""


def judge(task, expected, actual):
    prompt = JUDGE_PROMPT.format(task=task, expected=expected, actual=actual)
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": JUDGE_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
        timeout=60,
    ).json()
    raw = resp.get("response", "")

    passed = "VERDICT: PASS" in raw.upper() or "VERDICT:PASS" in raw.upper()
    reason = raw.split("REASON:")[-1].strip() if "REASON:" in raw else raw.strip()
    return passed, reason


def run_evals():
    baselines = metrics.all_baselines()
    if not baselines:
        print("No eval baselines saved yet. Use 'save eval' in a session after confirming a good answer.")
        return

    results = []
    for b in baselines:
        print(f"\n--- baseline #{b['id']}: {b['task'][:70]} ---")
        session = Session()
        actual = session.send(b["task"])
        passed, reason = judge(b["task"], b["expected_summary"], actual)
        metrics.log_eval_result(b["id"], passed, reason, actual)
        results.append(passed)
        print(f"VERDICT: {'PASS' if passed else 'FAIL'} -- {reason}")

    total = len(results)
    passed_count = sum(results)
    print(f"\n{passed_count}/{total} passed ({passed_count/total*100:.0f}%)")


if __name__ == "__main__":
    run_evals()
