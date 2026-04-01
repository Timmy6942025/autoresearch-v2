"""Report templates for different output formats."""

from __future__ import annotations

BRIEF_TEMPLATE = """# {title}

**Date:** {date}

## Summary

{summary}

## Key Points

{key_points}
"""

DETAILED_TEMPLATE = """# {title}

**Date:** {date}
**Sources:** {source_count}

## Executive Summary

{summary}

## Detailed Findings

{findings}

## Analysis

{analysis}

## Sources

{sources}
"""

PIPELINE_TEMPLATE = """# Research Pipeline: {name}

## Configuration
- Model: {model}
- TurboQuant: {turboquant}
- Context: {context}

## Steps

{steps}

## Results

{results}
"""
