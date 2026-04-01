"""Tests for the multi-agent research system."""

import asyncio
import json
from pathlib import Path

from autoresearch.agents.base import Agent, Message, Tool, ToolResult
from autoresearch.agents.planner import PlannerAgent
from autoresearch.agents.search_agent import SearchAgent
from autoresearch.agents.analyst import AnalystAgent
from autoresearch.agents.writer import WriterAgent
from autoresearch.core.config import ResearchConfig
from autoresearch.core.engine import ResearchEngine


class TestAgents:
    def test_planner_agent_creates_plan(self):
        agent = PlannerAgent()
        plan = agent.create_plan("Latest AI developments")
        assert "objective" in plan
        assert "subtasks" in plan
        assert len(plan["subtasks"]) >= 3

    def test_search_agent_initialization(self):
        agent = SearchAgent(max_results=5)
        assert agent.max_results == 5
        assert "web_search" in agent.tools

    def test_analyst_agent_initialization(self):
        agent = AnalystAgent()
        assert "analyze_data" in agent.tools

    def test_writer_agent_initialization(self):
        agent = WriterAgent()
        assert "generate_report" in agent.tools


class TestConfig:
    def test_default_config(self):
        config = ResearchConfig()
        assert config.search.max_results == 20
        assert config.crawler.concurrent == 8
        assert config.turboquant.enabled is False

    def test_config_from_dict(self):
        data = {
            "search": {"max_results": 5},
            "turboquant": {"enabled": True},
        }
        config = ResearchConfig.from_dict(data)
        assert config.search.max_results == 5
        assert config.turboquant.enabled is True

    def test_config_to_dict_roundtrip(self):
        config = ResearchConfig()
        data = config.to_dict()
        restored = ResearchConfig.from_dict(data)
        assert restored.search.max_results == config.search.max_results


class TestResearchEngine:
    def test_engine_initializes_agents(self):
        engine = ResearchEngine()
        assert "planner" in engine.agents
        assert "search" in engine.agents
        assert "crawler" in engine.agents
        assert "analyst" in engine.agents
        assert "writer" in engine.agents

    def test_engine_list_agents(self):
        engine = ResearchEngine()
        agents = engine.list_agents()
        assert len(agents) == 5

    def test_engine_get_agent(self):
        engine = ResearchEngine()
        planner = engine.get_agent("planner")
        assert planner is not None
        assert planner.name == "planner"

    def test_engine_with_custom_config(self):
        config = ResearchConfig()
        config.search.max_results = 5
        engine = ResearchEngine(config)
        assert engine.config.search.max_results == 5
