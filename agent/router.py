"""Task classifier + model router. Local mode uses Ollama for both the
classifier and the routed models -- fully offline, zero cost."""
from pathlib import Path

import requests
import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
CATEGORIES = ["investigate", "implement", "unit_tests", "extract", "general"]

CLASSIFY_PROMPT = (
    "Classify the following user message into exactly one category: "
    "investigate, implement, unit_tests, extract, or general. "
    "Use 'general' for greetings, casual conversation, or anything that "
    "is not a concrete coding task. "
    "Reply with ONLY the category word, nothing else.\n\n"
    "Message: {task}"
)


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class ModelRouter:
    def __init__(self, mode=None):
        self.config = load_config()
        self.mode = mode or self.config.get("mode", "local")
        router_cfg = self.config["router"][self.mode]
        self.classifier_model = router_cfg["classifier_model"]
        self.routing_table = router_cfg["routing_table"]

    def classify(self, task: str) -> str:
        prompt = CLASSIFY_PROMPT.format(task=task)
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.classifier_model, "prompt": prompt,
                    "stream": False, "options": {"temperature": 0},
                },
                timeout=30,
            ).json()
        except Exception:
            return "implement"  # safe default if classifier call fails

        raw = resp.get("response", "").strip().lower()
        for cat in CATEGORIES:
            if cat in raw or cat.replace("_", " ") in raw:
                return cat
        return "implement"

    def route(self, task: str) -> str:
        category = self.classify(task)
        model = self.routing_table.get(category, self.routing_table.get("implement"))
        return model, category
