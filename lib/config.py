"""Config abstraction layer."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Unified config management."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.getenv("HEARTHAGENT_CONFIG", "config.yaml")
        
        self.path = Path(config_path)
        self.mode = os.getenv("HEARTHAGENT_MODE", "local")
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load config from YAML."""
        if not self.path.exists():
            raise FileNotFoundError(f"Config not found: {self.path}")
        
        with open(self.path) as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot notation (e.g., 'router.local.classifier_model')."""
        keys = key.split(".")
        value = self.data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get config section for current mode."""
        return self.data.get(section, {}).get(self.mode, {})
