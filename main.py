#!/usr/bin/env python3
"""hearthagent-pro: local-first AI agent with model routing."""

import sys
import yaml
from pathlib import Path
from agent.router import ModelRouter


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    """Main entry point."""
    config = load_config()
    mode = config.get("mode", "local")

    print(f"🧠 hearthagent-pro ({mode} mode)")

    # Initialize router
    router = ModelRouter()
    print(f"Router loaded: {router.classifier_model}")

    # TODO: Initialize agent, memory, tools
    # TODO: Main loop

    print("Ready. (Implementation in progress)")


if __name__ == "__main__":
    main()
