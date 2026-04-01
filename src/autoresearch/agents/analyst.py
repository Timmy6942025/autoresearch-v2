"""Analyst agent — data analysis and insight extraction."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .base import Agent, Message, Tool, ToolResult

logger = logging.getLogger("autoresearch.agents.analyst")


class DataAnalysisTool(Tool):
    """Analyze text data for patterns and insights."""

    @property
    def name(self) -> str:
        return "analyze_data"

    @property
    def description(self) -> str:
        return "Analyze text data to extract key insights, patterns, and statistics."

    async def execute(self, data: str, analysis_type: str = "summary") -> ToolResult:
        try:
            if analysis_type == "summary":
                result = self._summarize(data)
            elif analysis_type == "keywords":
                result = self._extract_keywords(data)
            elif analysis_type == "sentiment":
                result = self._analyze_sentiment(data)
            else:
                result = {"error": f"Unknown analysis type: {analysis_type}"}

            return ToolResult(tool_name=self.name, result=result)
        except Exception as e:
            return ToolResult(tool_name=self.name, result=None, error=str(e))

    @staticmethod
    def _summarize(text: str) -> Dict[str, Any]:
        words = text.split()
        sentences = text.replace("!", ".").replace("?", ".").split(".")
        sentences = [s.strip() for s in sentences if s.strip()]

        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_sentence_length": round(len(words) / max(len(sentences), 1), 1),
            "unique_words": len(set(w.lower() for w in words)),
            "lexical_diversity": round(
                len(set(w.lower() for w in words)) / max(len(words), 1), 3
            ),
        }

    @staticmethod
    def _extract_keywords(text: str) -> Dict[str, Any]:
        words = text.lower().split()
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "i",
            "you",
            "he",
            "she",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
        }

        freq = {}
        for word in words:
            word = word.strip(".,!?;:\"'()[]{}")
            if word and word not in stop_words and len(word) > 2:
                freq[word] = freq.get(word, 0) + 1

        sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return {"keywords": sorted_keywords[:20]}

    @staticmethod
    def _analyze_sentiment(text: str) -> Dict[str, Any]:
        positive_words = {
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
            "awesome",
            "positive",
            "beneficial",
            "success",
            "successful",
            "improve",
            "improvement",
            "better",
            "best",
            "love",
            "like",
            "happy",
            "pleased",
            "satisfied",
        }
        negative_words = {
            "bad",
            "terrible",
            "awful",
            "horrible",
            "poor",
            "negative",
            "harmful",
            "fail",
            "failure",
            "worse",
            "worst",
            "hate",
            "dislike",
            "unhappy",
            "disappointed",
            "problem",
            "issue",
            "error",
            "wrong",
        }

        words = set(text.lower().split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        total = pos_count + neg_count

        if total == 0:
            sentiment = "neutral"
            score = 0.0
        else:
            score = (pos_count - neg_count) / total
            if score > 0.2:
                sentiment = "positive"
            elif score < -0.2:
                sentiment = "negative"
            else:
                sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "score": round(score, 2),
            "positive_words": pos_count,
            "negative_words": neg_count,
        }


class AnalystAgent(Agent):
    """Analyzes data and extracts insights from research findings."""

    def __init__(self):
        tools = [DataAnalysisTool()]
        system_prompt = """You are a data analyst agent. Your job is to analyze research findings and extract key insights.

When given data:
1. Use analyze_data tool to extract statistics
2. Identify key patterns and trends
3. Summarize findings in a structured format

Output format: JSON with keys: summary, key_insights, statistics, recommendations
"""
        super().__init__(name="analyst", system_prompt=system_prompt, tools=tools)

    async def process(self, message: Message) -> Message:
        self.add_message(message)

        try:
            data = json.loads(message.content)
        except json.JSONDecodeError:
            data = {"text": message.content, "analysis_type": "summary"}

        text = data.get("text", data.get("content", message.content))
        analysis_type = data.get("analysis_type", "summary")

        result = await self.use_tool(
            "analyze_data", data=text, analysis_type=analysis_type
        )

        if result.error:
            return Message(role="assistant", content=f"Analysis error: {result.error}")

        response = {
            "summary": f"Analysis complete. {result.result}",
            "key_insights": [],
            "statistics": result.result,
            "recommendations": ["Review the full data for deeper insights"],
        }

        if isinstance(result.result, dict) and "keywords" in result.result:
            response["key_insights"] = [
                f"Top keyword: {kw} ({count} occurrences)"
                for kw, count in result.result["keywords"][:5]
            ]

        return Message(role="assistant", content=json.dumps(response, indent=2))
