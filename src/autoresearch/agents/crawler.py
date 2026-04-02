"""Web crawler agent — async fetching and content extraction."""

from __future__ import annotations

import asyncio
import logging
import socket
from ipaddress import ip_address
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from .base import Agent, Message, Tool, ToolResult

logger = logging.getLogger("autoresearch.agents.crawler")


def _is_safe_url(url: str) -> bool:
    """Reject URLs pointing to private, loopback, or link-local addresses."""
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return False
    try:
        ip = ip_address(hostname)
    except ValueError:
        try:
            ip = ip_address(socket.gethostbyname(hostname))
        except (socket.gaierror, ValueError):
            return False
    return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved)


class WebCrawlerTool(Tool):
    """Fetch and extract content from web pages."""

    @property
    def name(self) -> str:
        return "web_crawl"

    @property
    def description(self) -> str:
        return "Fetch web pages and extract readable text content."

    async def execute(
        self,
        urls: List[str],
        max_concurrent: int = 5,
        timeout: int = 15,
    ) -> ToolResult:
        safe_urls = [u for u in urls if _is_safe_url(u)]
        blocked = [u for u in urls if not _is_safe_url(u)]
        if blocked:
            logger.warning("Blocked %d unsafe URLs: %s", len(blocked), blocked)

        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch(url: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        response = await client.get(url, follow_redirects=True)
                        response.raise_for_status()
                        content = self._extract_text(response.text)
                        return {
                            "url": url,
                            "status": response.status_code,
                            "title": self._extract_title(response.text),
                            "content": content[:5000],
                            "success": True,
                        }
                except Exception as e:
                    return {
                        "url": url,
                        "status": 0,
                        "title": "",
                        "content": "",
                        "success": False,
                        "error": str(e),
                    }

        tasks = [fetch(url) for url in urls]
        results = await asyncio.gather(*tasks)

        return ToolResult(
            tool_name=self.name,
            result=list(results),
            metadata={
                "total": len(urls),
                "successful": sum(1 for r in results if r["success"]),
            },
        )

    @staticmethod
    def _extract_text(html: str) -> str:
        """Extract readable text from HTML."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
        except ImportError:
            import re

            text = re.sub(r"<[^>]+>", " ", html)
            return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _extract_title(html: str) -> str:
        """Extract page title from HTML."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            title = soup.find("title")
            return title.get_text(strip=True) if title else ""
        except ImportError:
            import re

            match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
            return match.group(1).strip() if match else ""


class CrawlerAgent(Agent):
    """Fetches and extracts content from web pages."""

    def __init__(self, max_concurrent: int = 5, timeout: int = 15):
        tools = [WebCrawlerTool()]
        system_prompt = """You are a web crawler agent. Your job is to fetch web pages and extract readable content.

When given URLs:
1. Use the web_crawl tool to fetch them concurrently
2. Extract title and readable text content
3. Return structured results

Output format: JSON array of objects with keys: url, title, content, success
"""
        super().__init__(name="crawler", system_prompt=system_prompt, tools=tools)
        self.max_concurrent = max_concurrent
        self.timeout = timeout

    async def process(self, message: Message) -> Message:
        self.add_message(message)

        import json

        try:
            data = json.loads(message.content)
            urls = data.get("urls", [])
        except json.JSONDecodeError:
            urls = [message.content]

        if not urls:
            return Message(role="assistant", content="No URLs provided.")

        result = await self.use_tool(
            "web_crawl",
            urls=urls,
            max_concurrent=self.max_concurrent,
            timeout=self.timeout,
        )

        if result.error:
            return Message(role="assistant", content=f"Crawl error: {result.error}")

        return Message(role="assistant", content=json.dumps(result.result, indent=2))
