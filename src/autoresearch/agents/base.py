"""Base agent class with message passing and tool interface."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("autoresearch.agent")


@dataclass
class Message:
    """A message between agents or from agent to user."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Message":
        return cls(**d)


@dataclass
class ToolResult:
    """Result from a tool call."""
    tool_name: str
    result: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """A callable tool that agents can use."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        ...


class Agent(ABC):
    """Base class for research agents.

    Agents communicate via messages, can use tools, and maintain
    conversation history for context-aware responses.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str = "",
        tools: Optional[List[Tool]] = None,
        max_history: int = 50,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tools: Dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.history: List[Message] = []
        self.max_history = max_history

    def add_message(self, message: Message) -> None:
        self.history.append(message)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_context(self) -> str:
        """Build context string from system prompt and recent history."""
        parts = []
        if self.system_prompt:
            parts.append(f"System: {self.system_prompt}")
        for msg in self.history[-10:]:
            parts.append(f"{msg.role}: {msg.content}")
        return "\n".join(parts)

    async def use_tool(self, tool_name: str, **kwargs) -> ToolResult:
        tool = self.tools.get(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                result=None,
                error=f"Unknown tool: {tool_name}",
            )
        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                result=None,
                error=str(e),
            )

    @abstractmethod
    async def process(self, message: Message) -> Message:
        """Process an incoming message and return a response."""
        ...

    async def run(self, query: str) -> str:
        """Run the agent with a query and return the final result."""
        msg = Message(role="user", content=query)
        self.add_message(msg)
        response = await self.process(msg)
        self.add_message(response)
        return response.content
