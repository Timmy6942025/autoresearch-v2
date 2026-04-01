"""End-to-end integration tests for the multi-agent research system."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from autoresearch.agents.planner import PlannerAgent
from autoresearch.agents.search_agent import SearchAgent
from autoresearch.agents.crawler import CrawlerAgent
from autoresearch.agents.analyst import AnalystAgent
from autoresearch.agents.writer import WriterAgent
from autoresearch.core.config import ResearchConfig
from autoresearch.core.engine import ResearchEngine


class TestPlannerIntegration:
    def test_creates_structured_plan(self):
        agent = PlannerAgent()
        plan = agent.create_plan("What is quantum computing?")
        assert "objective" in plan
        assert "subtasks" in plan
        assert len(plan["subtasks"]) >= 3
        assert plan["complexity"] in ["low", "medium", "high"]

    def test_plan_has_agent_assignments(self):
        agent = PlannerAgent()
        plan = agent.create_plan("Latest AI developments 2025")
        agents_used = {s.get("agent") for s in plan["subtasks"]}
        assert "search" in agents_used


class TestSearchIntegration:
    def test_search_returns_results(self):
        agent = SearchAgent(max_results=5)
        result = asyncio.run(agent.run("Python programming"))
        data = json.loads(result) if isinstance(result, str) else result
        assert isinstance(data, list)

    def test_search_result_has_required_fields(self):
        agent = SearchAgent(max_results=3)
        result = asyncio.run(agent.run("machine learning basics"))
        data = json.loads(result) if isinstance(result, str) else result
        if data:
            first = data[0]
            assert "title" in first or "url" in first


class TestAnalystIntegration:
    def test_keyword_analysis(self):
        agent = AnalystAgent()
        result = asyncio.run(
            agent.run(
                json.dumps(
                    {
                        "text": "Python is a programming language. Python is used for machine learning. Python is popular.",
                        "analysis_type": "keywords",
                    }
                )
            )
        )
        data = json.loads(result) if isinstance(result, str) else result
        assert "statistics" in data or "key_insights" in data

    def test_summary_analysis(self):
        agent = AnalystAgent()
        result = asyncio.run(
            agent.run(
                json.dumps(
                    {
                        "text": "The quick brown fox jumps over the lazy dog. " * 10,
                        "analysis_type": "summary",
                    }
                )
            )
        )
        data = json.loads(result) if isinstance(result, str) else result
        assert "statistics" in data or "summary" in data


class TestWriterIntegration:
    def test_markdown_report(self):
        agent = WriterAgent()
        result = asyncio.run(
            agent.run(
                json.dumps(
                    {
                        "query": "Test topic",
                        "findings": [
                            {
                                "title": "Finding 1",
                                "url": "https://example.com",
                                "content": "Content here",
                            },
                        ],
                        "analysis": {"keywords": [["test", 5]]},
                        "format": "markdown",
                    }
                )
            )
        )
        assert "# Research Report" in result
        assert "Test topic" in result

    def test_json_report(self):
        agent = WriterAgent()
        result = asyncio.run(
            agent.run(
                json.dumps(
                    {
                        "query": "Test topic",
                        "findings": [{"title": "F1", "url": "", "content": "C1"}],
                        "analysis": {},
                        "format": "json",
                    }
                )
            )
        )
        data = json.loads(result)
        assert data["query"] == "Test topic"
        assert "findings" in data


class TestResearchEngineIntegration:
    def test_engine_full_pipeline(self):
        """Test the full research loop with a simple query."""
        config = ResearchConfig()
        config.search.max_results = 3
        config.crawler.concurrent = 2
        engine = ResearchEngine(config)

        report = asyncio.run(
            engine.research(
                query="What is Python programming language?",
                depth="brief",
                output_format="markdown",
            )
        )

        assert len(report) > 100
        assert "Python" in report or "programming" in report

    def test_engine_json_output(self):
        config = ResearchConfig()
        config.search.max_results = 3
        engine = ResearchEngine(config)

        report = asyncio.run(
            engine.research(
                query="What is MLX?",
                depth="brief",
                output_format="json",
            )
        )

        data = json.loads(report)
        assert "query" in data
        assert data["query"] == "What is MLX?"

    def test_engine_state_persistence(self):
        config = ResearchConfig()
        config.search.max_results = 3
        engine = ResearchEngine(config)

        asyncio.run(
            engine.research(
                query="Test query",
                depth="brief",
            )
        )

        state = engine.get_state()
        assert state.status == "completed"
        assert state.query == "Test query"
        assert state.report
