"""Search agent — DuckDuckGo integration for web research."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from .base import Agent, Message, Tool, ToolResult

logger = logging.getLogger("autoresearch.agents.search")


class DuckDuckGoSearchTool(Tool):
    """Search the web using DuckDuckGo."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web using DuckDuckGo. Returns title, URL, and snippet for each result."

    async def execute(self, query: str, max_results: int = 10) -> ToolResult:
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            return ToolResult(
                tool_name=self.name,
                result=results,
                metadata={"query": query, "count": len(results)},
            )
        except ImportError:
            return ToolResult(
                tool_name=self.name,
                result=None,
                error="duckduckgo-search not installed. Run: pip install duckduckgo-search",
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                result=None,
                error=str(e),
            )


class SearchAgent(Agent):
    """Searches the web and returns relevant results."""

    def __init__(self, max_results: int = 10):
        tools = [DuckDuckGoSearchTool()]
        system_prompt = f"""You are a web search agent. Your job is to find relevant information on the web.

When given a research query:
1. Use the web_search tool to find relevant results
2. Return up to {max_results} results with title, URL, and snippet
3. Format results as a structured list

Output format: JSON array of objects with keys: title, url, snippet
"""
        super().__init__(name="search", system_prompt=system_prompt, tools=tools)
        self.max_results = max_results

    async def process(self, message: Message) -> Message:
        self.add_message(message)

        query = message.content
        result = await self.use_tool(
            "web_search", query=query, max_results=self.max_results
        )

        if result.error:
            return Message(role="assistant", content=f"Search error: {result.error}")

        results = result.result or []
        formatted = []
        for r in results:
            formatted.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("url", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                }
            )

        response_content = json.dumps(formatted, indent=2)
        return Message(role="assistant", content=response_content)
