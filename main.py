import sys
from agent.graph import Session, run_task

BANNER = """
hearthagent-pro -- local agent with automatic model routing
type /quit to exit
"""


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
        session.send(user_input)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_task(" ".join(sys.argv[1:]))
    else:
        interactive_session()
