"""Model router: classify task type and route to optimal model."""

import yaml
from typing import Literal


class ModelRouter:
    """Task classifier + model router."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.mode = self.config.get("mode", "local")
        self.router_config = self.config["router"][self.mode]
        self.classifier_model = self.router_config["classifier_model"]
        self.routing_table = self.router_config["routing_table"]

    def classify_task(
        self, prompt: str
    ) -> Literal["investigate", "implement", "unit_tests", "extract"]:
        """Classify task type using classifier model."""
        # TODO: Call classifier_model with tight prompt
        # Returns one of: investigate, implement, unit_tests, extract
        return "investigate"

    def route(self, prompt: str) -> str:
        """Classify task and return optimal model."""
        task_type = self.classify_task(prompt)
        return self.routing_table[task_type]
