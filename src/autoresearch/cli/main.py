"""CLI interface for AutoResearch v2."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer

from ..core.config import ResearchConfig
from ..core.engine import ResearchEngine

app = typer.Typer(
    name="autoresearch",
    help="AutoResearch v2 — MLX-Powered Autonomous Research Engine",
    add_completion=False,
)

models_app = typer.Typer(name="models", help="Model management commands")
app.add_typer(models_app)


@models_app.command("list")
def models_list(
    downloaded_only: bool = typer.Option(
        False, "--downloaded", "-d", help="Show only downloaded models"
    ),
):
    """List available and downloaded models."""
    from ..core.models import registry

    typer.echo("Available Models:")
    typer.echo("-" * 60)
    for model in registry.list_all():
        status = "downloaded" if not downloaded_only else ""
        typer.echo(
            f"  {model.name:<20} {model.params_m:>6}M params  context={model.max_context:>6}  TQ={'✓' if model.supports_turboquant else '✗'}"
        )
    typer.echo("")


@models_app.command("pull")
def models_pull(
    name: str = typer.Argument(..., help="Model name or hub ID"),
):
    """Download a model from MLX hub."""
    from ..core.models import registry, ModelInfo

    existing = registry.get(name)
    if existing:
        typer.echo(f"Model '{name}' already registered: {existing.path}")
        return

    typer.echo(f"Registering model: {name}")
    typer.echo("Note: Actual download requires mlx-lm. Run:")
    typer.echo(f"  python -c 'from mlx_lm import load; load(\"{name}\")'")


@models_app.command("remove")
def models_remove(
    name: str = typer.Argument(..., help="Model name to remove"),
):
    """Delete a downloaded model."""
    from ..core.models import registry

    if registry.remove(name):
        typer.echo(f"Removed model: {name}")
    else:
        typer.echo(f"Model not found: {name}")


@models_app.command("info")
def models_info(
    name: str = typer.Argument(..., help="Model name"),
):
    """Show model details."""
    from ..core.models import registry

    model = registry.get(name)
    if not model:
        typer.echo(f"Model not found: {name}")
        return

    typer.echo(f"Model: {model.name}")
    typer.echo(f"Path: {model.path}")
    typer.echo(f"Parameters: {model.params_m}M")
    typer.echo(f"Max Context: {model.max_context}")
    typer.echo(f"TurboQuant: {'Yes' if model.supports_turboquant else 'No'}")
    typer.echo(f"Quantization: {model.quant_level}")


@models_app.command("benchmark")
def models_benchmark():
    """Benchmark all downloaded models."""
    typer.echo("Benchmark mode — requires MLX models loaded")
    typer.echo("Run with --model to benchmark a specific model")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
    port: int = typer.Option(8080, "--port", help="Port number"),
    model: Optional[str] = typer.Option(None, "--model", help="Model to serve"),
    turboquant: bool = typer.Option(False, "--turboquant", help="Enable compression"),
    cors: bool = typer.Option(True, "--cors/--no-cors", help="Enable CORS"),
):
    """Run as an API server for external integrations."""
    typer.echo(f"Starting server on {host}:{port}")
    typer.echo(f"Model: {model or 'default'}")
    typer.echo(f"TurboQuant: {'enabled' if turboquant else 'disabled'}")
    typer.echo(f"CORS: {'enabled' if cors else 'disabled'}")
    typer.echo("\nNote: Requires fastapi and uvicorn.")
    typer.echo("  pip install fastapi uvicorn")


@app.command()
def pipeline(
    spec: str = typer.Argument(..., help="Pipeline spec file (YAML/JSON)"),
):
    """Run a predefined research pipeline from a JSON/YAML spec."""
    path = Path(spec)
    if not path.exists():
        typer.echo(f"Spec file not found: {spec}", err=True)
        raise typer.Exit(1)

    try:
        if path.suffix in [".yaml", ".yml"]:
            import yaml

            data = yaml.safe_load(path.read_text())
        else:
            data = json.loads(path.read_text())
    except Exception as e:
        typer.echo(f"Failed to parse spec: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Pipeline: {data.get('name', 'unnamed')}")
    typer.echo(f"Model: {data.get('model', 'default')}")
    typer.echo(f"Steps: {len(data.get('steps', []))}")

    config = ResearchConfig()
    if "model" in data:
        config.model.path = data["model"]
    if data.get("turboquant"):
        config.turboquant.enabled = True

    engine = ResearchEngine(config)

    async def run_pipeline():
        for i, step in enumerate(data.get("steps", []), 1):
            if "search" in step:
                typer.echo(f"Step {i}: Searching '{step['search']}'")
                max_sources = step.get("max_sources", 10)
                config.search.max_results = max_sources
            elif "analyze" in step:
                typer.echo(f"Step {i}: Analyzing — {step['analyze']}")
            elif "synthesize" in step:
                typer.echo(f"Step {i}: Synthesizing → {step['synthesize']}")

    asyncio.run(run_pipeline())
    typer.echo("Pipeline complete.")


@app.command()
def research(
    query: str = typer.Argument(..., help="Research query"),
    model: Optional[str] = typer.Option(None, "--model", help="MLX model to use"),
    depth: str = typer.Option(
        "medium", "--depth", help="Research depth: brief, medium, deep"
    ),
    turboquant: bool = typer.Option(
        False, "--turboquant", help="Enable KV cache compression"
    ),
    context_window: str = typer.Option(
        "32k", "--context-window", help="Max context: 32k, 64k, 128k"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    format: str = typer.Option("md", "--format", help="Output format: md, html, json"),
    max_iterations: Optional[int] = typer.Option(
        None, "--max-iterations", help="Max research iterations"
    ),
    max_sources: int = typer.Option(20, "--max-sources", help="Max sources to analyze"),
    timeout: int = typer.Option(600, "--timeout", help="Max runtime in seconds"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed logs"),
):
    """Run autonomous research on a query."""
    import logging

    if verbose:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
        )

    config = ResearchConfig()
    if model:
        config.model.path = model
    if turboquant:
        config.turboquant.enabled = True
    config.search.max_results = max_sources

    engine = ResearchEngine(config)

    output_format = "markdown" if format == "md" else format

    typer.echo(f"Starting research: '{query}'")
    typer.echo(f"Depth: {depth} | Format: {output_format}")

    try:
        report = asyncio.run(
            engine.research(
                query=query,
                depth=depth,
                max_iterations=max_iterations,
                output_path=output,
                output_format=output_format,
            )
        )

        if not output:
            typer.echo("\n" + "=" * 60)
            typer.echo(report)
        else:
            typer.echo(f"\nReport saved to: {output}")

    except KeyboardInterrupt:
        typer.echo("\nResearch interrupted.")
        sys.exit(1)
    except Exception as e:
        typer.echo(f"\nError: {e}", err=True)
        sys.exit(1)


@app.command()
def interactive(
    model: Optional[str] = typer.Option(None, "--model", help="Model to use"),
    turboquant: bool = typer.Option(False, "--turboquant", help="Enable compression"),
    history: bool = typer.Option(
        True, "--history/--no-history", help="Enable persistent chat history"
    ),
    system_prompt: Optional[str] = typer.Option(
        None, "--system-prompt", help="Custom system prompt"
    ),
):
    """Interactive REPL mode with research capabilities."""
    from ..agents.base import Agent, Message
    from ..agents.planner import PlannerAgent
    from ..agents.search_agent import SearchAgent
    from ..agents.writer import WriterAgent

    config = ResearchConfig()
    if model:
        config.model.path = model

    planner = PlannerAgent()
    search = SearchAgent(max_results=config.search.max_results)
    writer = WriterAgent()

    typer.echo("AutoResearch v2 Interactive Mode")
    typer.echo("Type /help for commands, /quit to exit")
    typer.echo("")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            command = user_input[1:].strip().split()[0]
            args = user_input[1:].strip().split()[1:]

            if command == "quit" or command == "exit":
                typer.echo("Goodbye!")
                break
            elif command == "help":
                typer.echo("Commands:")
                typer.echo("  /research QUERY    Run deep research")
                typer.echo("  /search QUERY      Quick web search")
                typer.echo("  /plan QUERY        Create research plan")
                typer.echo("  /quit              Exit")
                typer.echo("  /help              Show this help")
            elif command == "research":
                query = " ".join(args)
                if query:
                    typer.echo("Planning...")
                    plan = asyncio.run(planner.run(query))
                    typer.echo("\nPlan:")
                    typer.echo(plan)

                    typer.echo("\nSearching...")
                    search_result = asyncio.run(search.run(query))
                    typer.echo("\nSearch Results:")
                    typer.echo(search_result)
            elif command == "search":
                query = " ".join(args)
                if query:
                    result = asyncio.run(search.run(query))
                    typer.echo(result)
            elif command == "plan":
                query = " ".join(args)
                if query:
                    plan = asyncio.run(planner.run(query))
                    typer.echo(plan)
            else:
                typer.echo(f"Unknown command: /{command}")
        else:
            typer.echo("Use /research, /search, /plan, or /help")


@app.command()
def benchmark(
    model: Optional[str] = typer.Option(None, "--model", help="Model to benchmark"),
    prompt_length: int = typer.Option(4096, "--prompt-length", help="Input tokens"),
    gen_length: int = typer.Option(256, "--gen-length", help="Output tokens"),
    turboquant: bool = typer.Option(
        False, "--turboquant", help="With KV cache compression"
    ),
    warmup: int = typer.Option(3, "--warmup", help="Warmup runs"),
    runs: int = typer.Option(5, "--runs", help="Benchmark runs"),
):
    """Benchmark model performance on your hardware."""
    typer.echo("Benchmark mode — requires MLX model")
    typer.echo(f"Model: {model or 'default'}")
    typer.echo(f"Prompt: {prompt_length} tokens, Generation: {gen_length} tokens")
    typer.echo(f"Runs: {runs} (warmup: {warmup})")
    typer.echo("\nNote: MLX benchmarking requires actual model loading.")
    typer.echo("This is a placeholder — full implementation needs MLX backend.")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
