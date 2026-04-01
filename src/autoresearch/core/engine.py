"""Research engine — orchestrates the multi-agent research loop."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..agents.base import Agent, Message
from ..agents.planner import PlannerAgent
from ..agents.search_agent import SearchAgent
from ..agents.crawler import CrawlerAgent
from ..agents.analyst import AnalystAgent
from ..agents.writer import WriterAgent
from .config import ResearchConfig

logger = logging.getLogger("autoresearch.engine")


class ResearchEngine:
    """Orchestrates the multi-agent research loop.

    The engine coordinates multiple specialized agents to:
    1. Plan research strategy
    2. Search for information
    3. Crawl and extract content
    4. Analyze findings
    5. Synthesize reports
    """

    def __init__(self, config: Optional[ResearchConfig] = None):
        self.config = config or ResearchConfig()
        self.agents: Dict[str, Agent] = {}
        self._initialize_agents()

    def _initialize_agents(self) -> None:
        """Create and register all research agents."""
        self.agents["planner"] = PlannerAgent()
        self.agents["search"] = SearchAgent(max_results=self.config.search.max_results)
        self.agents["crawler"] = CrawlerAgent(
            max_concurrent=self.config.crawler.concurrent,
            timeout=self.config.crawler.timeout,
        )
        self.agents["analyst"] = AnalystAgent()
        self.agents["writer"] = WriterAgent()

    async def research(
        self,
        query: str,
        depth: str = "medium",
        max_iterations: Optional[int] = None,
        output_path: Optional[str] = None,
        output_format: str = "markdown",
    ) -> str:
        """Execute a full research loop.

        Args:
            query: The research query
            depth: Research depth (brief, medium, deep)
            max_iterations: Maximum research iterations
            output_path: Path to save the report
            output_format: Output format (markdown, json)

        Returns:
            The generated research report
        """
        iterations = max_iterations or {
            "brief": 3,
            "medium": 7,
            "deep": 15,
        }.get(depth, 7)

        logger.info(
            f"Starting research: '{query}' (depth={depth}, iterations={iterations})"
        )

        # Step 1: Plan
        logger.info("Step 1: Planning research strategy")
        plan_result = await self.agents["planner"].run(query)
        try:
            plan = json.loads(plan_result)
        except json.JSONDecodeError:
            plan = {"objective": query, "subtasks": []}

        # Step 2: Search
        logger.info("Step 2: Searching for information")
        search_result = await self.agents["search"].run(query)
        try:
            search_results = json.loads(search_result)
        except json.JSONDecodeError:
            search_results = []

        # Step 3: Crawl
        logger.info("Step 3: Crawling relevant pages")
        urls = [
            r.get("url", r.get("href", ""))
            for r in search_results
            if r.get("url") or r.get("href")
        ]

        if urls:
            crawl_input = json.dumps({"urls": urls[:10]})
            crawl_result = await self.agents["crawler"].run(crawl_input)
            try:
                crawled_content = json.loads(crawl_result)
            except json.JSONDecodeError:
                crawled_content = []
        else:
            crawled_content = []

        # Step 4: Analyze
        logger.info("Step 4: Analyzing findings")
        all_content = []
        for item in search_results:
            all_content.append(item.get("snippet", item.get("body", "")))
        for item in crawled_content:
            if item.get("success"):
                all_content.append(item.get("content", ""))

        combined_text = "\n\n".join(all_content)
        if combined_text:
            analysis_input = json.dumps(
                {
                    "text": combined_text[:10000],
                    "analysis_type": "keywords",
                }
            )
            analysis_result = await self.agents["analyst"].run(analysis_input)
            try:
                analysis = json.loads(analysis_result)
            except json.JSONDecodeError:
                analysis = {}
        else:
            analysis = {}

        # Step 5: Synthesize
        logger.info("Step 5: Synthesizing report")
        findings = []
        for item in search_results:
            findings.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", item.get("href", "")),
                    "content": item.get("snippet", item.get("body", "")),
                }
            )

        report_input = json.dumps(
            {
                "query": query,
                "findings": findings,
                "analysis": analysis,
                "format": output_format,
            }
        )

        report = await self.agents["writer"].run(report_input)

        # Save report
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report)
            logger.info(f"Report saved to: {output_path}")

        logger.info("Research complete")
        return report

    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name."""
        return self.agents.get(name)

    def list_agents(self) -> List[str]:
        """List all registered agents."""
        return list(self.agents.keys())
