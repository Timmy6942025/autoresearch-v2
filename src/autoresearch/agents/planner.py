"""Research planner agent — decomposes queries into research plans."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .base import Agent, Message, Tool, ToolResult


class PlannerAgent(Agent):
    """Decomposes research queries into structured plans with subtasks."""

    def __init__(self, tools: Optional[List[Tool]] = None):
        system_prompt = """You are a research planner. Your job is to decompose research queries into structured plans.

For each query, produce:
1. A clear research objective
2. 3-5 subtasks that need to be completed
3. Priority ordering for the subtasks
4. Estimated complexity (low/medium/high)

Output format: JSON with keys: objective, subtasks, priority_order, complexity
"""
        super().__init__(name="planner", system_prompt=system_prompt, tools=tools)

    async def process(self, message: Message) -> Message:
        self.add_message(message)
        
        # In a real implementation, this would call an LLM
        # For now, we'll create a structured plan based on the query
        query = message.content
        
        plan = {
            "objective": f"Research: {query}",
            "subtasks": [
                {"id": 1, "task": f"Search for information about '{query}'", "agent": "search"},
                {"id": 2, "task": "Crawl and extract relevant web pages", "agent": "crawler"},
                {"id": 3, "task": "Analyze findings and extract key insights", "agent": "analyst"},
                {"id": 4, "task": "Synthesize results into a comprehensive report", "agent": "writer"},
            ],
            "priority_order": [1, 2, 3, 4],
            "complexity": "medium",
        }
        
        response_content = json.dumps(plan, indent=2)
        return Message(role="assistant", content=response_content)

    def create_plan(self, query: str) -> Dict[str, Any]:
        """Synchronous plan creation for simple cases."""
        return {
            "objective": f"Research: {query}",
            "subtasks": [
                {"id": 1, "task": f"Search for information about '{query}'", "agent": "search"},
                {"id": 2, "task": "Crawl and extract relevant web pages", "agent": "crawler"},
                {"id": 3, "task": "Analyze findings and extract key insights", "agent": "analyst"},
                {"id": 4, "task": "Synthesize results into a comprehensive report", "agent": "writer"},
            ],
            "priority_order": [1, 2, 3, 4],
            "complexity": "medium",
        }
