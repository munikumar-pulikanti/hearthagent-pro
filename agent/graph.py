"""LangGraph ReAct agent with tool calling."""

import json
from typing import Any, Dict, List


class ReActAgent:
    """LangGraph-based ReAct agent."""

    def __init__(self, model: str, tools: List[Any] = None):
        self.model = model
        self.tools = tools or []
        self.memory = []

    def think(self, prompt: str) -> str:
        """Generate thought via LLM."""
        # TODO: Call LLM with prompt
        return "Thinking..."

    def act(self, action: str) -> str:
        """Execute tool action."""
        # TODO: Match action to tool, execute
        return "Action result"

    def observe(self, result: str):
        """Store observation in memory."""
        self.memory.append({"type": "observation", "content": result})

    def run(self, prompt: str, max_steps: int = 10) -> str:
        """Run ReAct loop: Thought → Action → Observation → ..."""
        for step in range(max_steps):
            thought = self.think(prompt)
            self.memory.append({"type": "thought", "content": thought})
            
            # Parse action from thought
            # TODO: Extract action from LLM response
            
            result = self.act("placeholder")
            self.observe(result)
            
            # TODO: Check if done
        
        # Return final answer
        return "Final answer"
