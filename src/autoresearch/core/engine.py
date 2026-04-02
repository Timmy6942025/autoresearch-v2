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
from ..documents.parser import DocumentParser, DocumentChunker
from ..scheduler.dispatcher import TaskDispatcher, Task
from .config import ResearchConfig
from .state import ResearchState, StateManager, CacheManager

logger = logging.getLogger("autoresearch.engine")


class ResearchEngine:
    """Orchestrates the multi-agent research loop.

    Full pipeline:
    1. Planner decomposes query into subtasks
    2. Search agents run parallel queries
    3. Crawler fetches and extracts content
    4. Analyst extracts insights from all content
    5. Writer synthesizes comprehensive report
    6. State persisted, results cached
    """

    def __init__(self, config: Optional[ResearchConfig] = None):
        self.config = config or ResearchConfig()
        self.agents: Dict[str, Agent] = {}
        self.state = ResearchState()
        self.state_manager = StateManager()
        self.cache = CacheManager(
            cache_dir=self.config.cache.directory,
            max_size_mb=self.config.cache.max_size_mb,
        )
        self.dispatcher = TaskDispatcher(
            max_workers=self.config.crawler.concurrent,
        )
        self._initialize_agents()

    def _initialize_agents(self) -> None:
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
        files: Optional[List[str]] = None,
    ) -> str:
        """Execute a full multi-agent research loop."""
        iterations = max_iterations or {"brief": 3, "medium": 7, "deep": 15}.get(
            depth, 7
        )

        self.state = ResearchState(
            query=query,
            status="running",
            started_at=time.time(),
        )

        logger.info(
            "Starting research: '%s' (depth=%s, iterations=%d)",
            query,
            depth,
            iterations,
        )

        # Step 1: Plan — decompose query into subtasks
        logger.info("Step 1: Planning research strategy")
        plan_result = await self.agents["planner"].run(query)
        try:
            plan = json.loads(plan_result)
        except json.JSONDecodeError:
            plan = {"objective": query, "subtasks": []}

        self.state.metadata["plan"] = plan

        # Step 2: Parallel search — run multiple search queries from plan
        logger.info("Step 2: Searching for information")
        search_queries = [query]
        for subtask in plan.get("subtasks", []):
            if subtask.get("agent") == "search":
                search_queries.append(subtask["task"])

        search_tasks = [
            self.agents["search"].run(q)
            for q in search_queries[: self.config.agent.parallel_searches]
        ]
        search_results_raw = await asyncio.gather(*search_tasks, return_exceptions=True)

        search_results = []
        for raw in search_results_raw:
            if isinstance(raw, Exception):
                logger.warning("Search failed: %s", raw)
                continue
            try:
                results = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(results, list):
                    search_results.extend(results)
            except json.JSONDecodeError:
                pass

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in search_results:
            url = r.get("url", r.get("href", ""))
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)
        search_results = unique_results[: self.config.search.max_results]

        # Step 3: Crawl — fetch content from search results in parallel
        logger.info("Step 3: Crawling relevant pages (%d URLs)", len(search_results))
        urls = [
            r.get("url", r.get("href", ""))
            for r in search_results
            if r.get("url") or r.get("href")
        ]

        crawled_content = []
        if urls:
            crawl_input = json.dumps({"urls": urls[:10]})
            crawl_result = await self.agents["crawler"].run(crawl_input)
            try:
                crawled_content = (
                    json.loads(crawl_result)
                    if isinstance(crawl_result, str)
                    else crawl_result
                )
            except json.JSONDecodeError:
                crawled_content = []

        # Step 3b: Process local documents if provided
        document_content = []
        if files:
            allowed_dir = Path.cwd().resolve()
            logger.info("Step 3b: Processing %d local documents", len(files))
            for file_path in files:
                resolved = Path(file_path).resolve()
                if not str(resolved).startswith(str(allowed_dir)):
                    logger.warning("Blocked path traversal attempt: %s", file_path)
                    continue
                result = DocumentParser.parse(file_path)
                if result.get("success"):
                    chunks = DocumentChunker.chunk(
                        result.get("content", ""), max_chunk_size=2000
                    )
                    for i, chunk in enumerate(chunks):
                        document_content.append(
                            {
                                "source": file_path,
                                "chunk": i,
                                "title": result.get("title", file_path),
                                "content": chunk,
                            }
                        )

        # Step 4: Analyze — extract insights from all content
        logger.info("Step 4: Analyzing findings")
        all_text_parts = []
        for item in search_results:
            all_text_parts.append(item.get("snippet", item.get("body", "")))
        for item in crawled_content:
            if isinstance(item, dict) and item.get("success"):
                all_text_parts.append(item.get("content", ""))
        for item in document_content:
            all_text_parts.append(item.get("content", ""))

        combined_text = "\n\n".join(all_text_parts)
        analysis = {}
        if combined_text:
            analysis_input = json.dumps(
                {
                    "text": combined_text[:15000],
                    "analysis_type": "keywords",
                }
            )
            analysis_result = await self.agents["analyst"].run(analysis_input)
            try:
                analysis = (
                    json.loads(analysis_result)
                    if isinstance(analysis_result, str)
                    else analysis_result
                )
            except json.JSONDecodeError:
                analysis = {}

        self.state.analysis = analysis
        self.state.findings = search_results

        # Step 5: Synthesize — generate comprehensive report
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
        for item in crawled_content:
            if isinstance(item, dict) and item.get("success"):
                findings.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "content": item.get("content", "")[:2000],
                    }
                )
        for item in document_content:
            findings.append(
                {
                    "title": f"[Document] {item.get('title', '')}",
                    "url": "",
                    "content": item.get("content", ""),
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
        self.state.report = report
        self.state.status = "completed"
        self.state.completed_at = time.time()

        # Persist state
        self.state_manager.save(self.state)

        # Save report to file
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report)
            logger.info("Report saved to: %s", output_path)

        # Cache the result
        if self.config.cache.enabled:
            cache_key = f"research:{query}:{depth}"
            self.cache.set(
                cache_key, {"report": report, "findings": findings}, ttl=86400
            )

        logger.info("Research complete")
        return report

    def get_agent(self, name: str) -> Optional[Agent]:
        return self.agents.get(name)

    def list_agents(self) -> List[str]:
        return list(self.agents.keys())

    def get_state(self) -> ResearchState:
        return self.state
