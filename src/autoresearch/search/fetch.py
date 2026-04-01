"""URL content fetching."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("autoresearch.search.fetch")


async def fetch_url(
    url: str,
    timeout: int = 15,
    user_agent: str = "AutoResearchBot/2.0",
) -> Dict[str, Any]:
    """Fetch a URL and return content."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                url,
                follow_redirects=True,
                headers={"User-Agent": user_agent},
            )
            response.raise_for_status()
            return {
                "url": url,
                "status": response.status_code,
                "content": response.text,
                "content_type": response.headers.get("content-type", ""),
                "success": True,
            }
    except Exception as e:
        return {
            "url": url,
            "status": 0,
            "content": "",
            "success": False,
            "error": str(e),
        }


async def fetch_urls(
    urls: List[str],
    max_concurrent: int = 8,
    timeout: int = 15,
) -> List[Dict[str, Any]]:
    """Fetch multiple URLs concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_one(url: str) -> Dict[str, Any]:
        async with semaphore:
            return await fetch_url(url, timeout=timeout)

    tasks = [fetch_one(url) for url in urls]
    return await asyncio.gather(*tasks, return_exceptions=True)
