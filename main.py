import subprocess
import sys

from agent.graph import Session, run_task
from agent import metrics

BANNER = """
hearthagent-pro -- local agent with automatic model routing
type /quit to exit | type 'save eval' after confirming an answer is correct
"""


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def save_last_as_eval(session):
    if len(session.history) < 2:
        print("Nothing to save yet -- have at least one exchange first.")
        return

    last_user = None
    last_assistant = None
    for m in reversed(session.history):
        role = getattr(m, "type", None) or (m[0] if isinstance(m, tuple) else None)
        content = getattr(m, "content", None) or (m[1] if isinstance(m, tuple) else None)
        if role in ("ai", "assistant") and last_assistant is None:
            last_assistant = content
        if role in ("human", "user") and last_user is None and last_assistant is not None:
            last_user = content
        if last_user and last_assistant:
            break

    if not last_user or not last_assistant:
        print("Could not find a complete task/answer pair to save.")
        return

    commit = get_git_commit()
    baseline_id = metrics.save_eval_baseline(last_user, last_assistant, commit)
    print(f"--- saved eval baseline #{baseline_id} (git commit: {commit or 'none'}) ---")


def interactive_session():
    print(BANNER)
    session = Session()
    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user_input:
            continue
        if user_input in ("/quit", "/exit"):
            print("Exiting.")
            break
        if user_input.lower() == "save eval":
            save_last_as_eval(session)
            continue
        session.send(user_input)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_task(" ".join(sys.argv[1:]))
    else:
        interactive_session()
