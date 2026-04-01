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
    model_path = existing.path if existing else name

    typer.echo(f"Downloading model: {model_path}")
    typer.echo("This may take a while depending on model size and network speed...")

    try:
        from mlx_lm import load

        typer.echo("Loading model (this downloads if not cached)...")
        model, tokenizer = load(model_path)
        typer.echo(f"Model loaded successfully: {model_path}")

        if not existing:
            registry.register(
                ModelInfo(
                    name=name,
                    path=model_path,
                    params_m=0,
                    max_context=32768,
                    supports_turboquant=True,
                )
            )
            typer.echo(f"Registered as: {name}")
    except ImportError:
        typer.echo("mlx-lm not installed. Install with: pip install mlx-lm")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Failed to download/load model: {e}", err=True)
        raise typer.Exit(1)


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
    from ..server import run_server

    run_server(host=host, port=port, enable_cors=cors)


@app.command()
def pipeline(
    spec: str = typer.Argument(..., help="Pipeline spec file (YAML/JSON)"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output report path"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed logs"),
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
    all_findings = []
    all_content = []

    async def run_pipeline():
        for i, step in enumerate(data.get("steps", []), 1):
            if "search" in step:
                query = step["search"]
                max_sources = step.get("max_sources", 10)
                typer.echo(
                    f"\nStep {i}: Searching '{query}' (max {max_sources} sources)"
                )
                config.search.max_results = max_sources
                search_result = await engine.agents["search"].run(query)
                try:
                    results = (
                        json.loads(search_result)
                        if isinstance(search_result, str)
                        else search_result
                    )
                    all_findings.extend(results)
                    for r in results:
                        all_content.append(r.get("snippet", r.get("body", "")))
                    typer.echo(f"  Found {len(results)} results")
                except json.JSONDecodeError:
                    typer.echo("  Search returned no results")

                urls = [
                    r.get("url", r.get("href", ""))
                    for r in (results if isinstance(results, list) else [])
                    if r.get("url") or r.get("href")
                ]
                if urls:
                    typer.echo(f"  Crawling {min(len(urls), 5)} pages...")
                    crawl_result = await engine.agents["crawler"].run(
                        json.dumps({"urls": urls[:5]})
                    )
                    try:
                        crawled = (
                            json.loads(crawl_result)
                            if isinstance(crawl_result, str)
                            else crawl_result
                        )
                        for c in crawled if isinstance(crawled, list) else []:
                            if c.get("success"):
                                all_content.append(c.get("content", "")[:2000])
                    except json.JSONDecodeError:
                        pass

            elif "analyze" in step:
                instruction = step["analyze"]
                typer.echo(f"\nStep {i}: Analyzing — {instruction}")
                combined = "\n\n".join(all_content[:10])
                if combined:
                    analysis = await engine.agents["analyst"].run(
                        json.dumps(
                            {
                                "text": combined[:10000],
                                "analysis_type": "keywords",
                            }
                        )
                    )
                    typer.echo(f"  Analysis complete")
                else:
                    typer.echo("  No content to analyze (run search steps first)")

            elif "synthesize" in step:
                output_file = step["synthesize"]
                typer.echo(f"\nStep {i}: Synthesizing → {output_file}")
                findings = [
                    {
                        "title": f.get("title", ""),
                        "url": f.get("url", f.get("href", "")),
                        "content": f.get("snippet", f.get("body", "")),
                    }
                    for f in all_findings
                ]
                report = synthesizer.synthesize(
                    query=data.get("name", "Research"),
                    findings=findings,
                    format="markdown",
                )
                out_path = Path(output_file)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(report)
                typer.echo(f"  Report saved to: {output_file}")

    from ..synthesis.synthesizer import Synthesizer

    synthesizer = Synthesizer()
    asyncio.run(run_pipeline())
    typer.echo("\nPipeline complete.")


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
    import json
    import time
    from datetime import datetime
    from pathlib import Path

    from ..agents.analyst import AnalystAgent
    from ..agents.crawler import CrawlerAgent
    from ..agents.planner import PlannerAgent
    from ..agents.search_agent import SearchAgent
    from ..agents.writer import WriterAgent
    from ..core.state import StateManager
    from ..documents.parser import DocumentParser
    from ..synthesis.synthesizer import Synthesizer

    config = ResearchConfig()
    if model:
        config.model.path = model
    if turboquant:
        config.turboquant.enabled = True

    planner = PlannerAgent()
    search = SearchAgent(max_results=config.search.max_results)
    crawler = CrawlerAgent()
    analyst = AnalystAgent()
    writer = WriterAgent()
    synthesizer = Synthesizer()
    state_mgr = StateManager()

    context_window = 32768
    chat_history: list[dict] = []
    current_session_id = f"session_{int(time.time())}"

    typer.echo("AutoResearch v2 Interactive Mode")
    typer.echo(
        f"Context: {context_window} tokens | TurboQuant: {'on' if turboquant else 'off'}"
    )
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
            parts = user_input[1:].strip().split()
            command = parts[0]
            args = parts[1:]

            if command in ("quit", "exit"):
                if history and chat_history:
                    state_mgr.save(
                        type(
                            "State",
                            (),
                            {
                                "query": "interactive",
                                "status": "completed",
                                "started_at": time.time(),
                                "completed_at": time.time(),
                                "findings": [],
                                "analysis": {},
                                "report": json.dumps(chat_history[-5:]),
                                "error": None,
                                "metadata": {},
                            },
                        )(),
                        current_session_id,
                    )
                typer.echo("Goodbye!")
                break

            elif command == "help":
                typer.echo("Commands:")
                typer.echo("  /research QUERY        Run deep research")
                typer.echo("  /search QUERY          Quick web search")
                typer.echo("  /plan QUERY            Create research plan")
                typer.echo("  /summarize URL         Summarize a web page")
                typer.echo("  /analyze FILE/TEXT     Analyze a document or text")
                typer.echo("  /compare TOPIC1 vs TOPIC2  Compare two topics")
                typer.echo("  /turboquant [on|off|status]  Toggle compression")
                typer.echo("  /context [32k|64k|128k]  Set context window")
                typer.echo("  /model NAME            Switch models")
                typer.echo("  /save FILENAME         Save conversation")
                typer.echo("  /history               Show recent history")
                typer.echo("  /clear                 Clear chat history")
                typer.echo("  /quit                  Exit")
                typer.echo("  /help                  Show this help")

            elif command == "research":
                query = " ".join(args)
                if not query:
                    typer.echo("Usage: /research QUERY")
                    continue
                typer.echo("Planning...")
                plan = asyncio.run(planner.run(query))
                typer.echo(f"\nPlan:\n{plan}")
                typer.echo("\nSearching...")
                search_result = asyncio.run(search.run(query))
                typer.echo(f"\nSearch Results:\n{search_result}")

            elif command == "search":
                query = " ".join(args)
                if not query:
                    typer.echo("Usage: /search QUERY")
                    continue
                result = asyncio.run(search.run(query))
                typer.echo(result)

            elif command == "plan":
                query = " ".join(args)
                if not query:
                    typer.echo("Usage: /plan QUERY")
                    continue
                plan = asyncio.run(planner.run(query))
                typer.echo(plan)

            elif command == "summarize":
                url = " ".join(args)
                if not url:
                    typer.echo("Usage: /summarize URL")
                    continue
                typer.echo(f"Fetching {url}...")
                crawl_result = asyncio.run(crawler.run(json.dumps({"urls": [url]})))
                try:
                    data = (
                        json.loads(crawl_result)
                        if isinstance(crawl_result, str)
                        else crawl_result
                    )
                    if data and isinstance(data, list) and data[0].get("success"):
                        content = data[0].get("content", "")[:5000]
                        title = data[0].get("title", url)
                        typer.echo(f"\nSummary of: {title}\n")
                        typer.echo(content[:2000])
                    else:
                        typer.echo(f"Failed to fetch: {url}")
                except (json.JSONDecodeError, IndexError):
                    typer.echo(f"Failed to parse response for: {url}")

            elif command == "analyze":
                target = " ".join(args)
                if not target:
                    typer.echo("Usage: /analyze FILE or /analyze TEXT")
                    continue
                path = Path(target)
                if path.exists():
                    typer.echo(f"Analyzing document: {target}")
                    result = DocumentParser.parse(path)
                    if result.get("success"):
                        content = result.get("content", "")[:5000]
                        analysis = asyncio.run(
                            analyst.run(
                                json.dumps(
                                    {
                                        "text": content,
                                        "analysis_type": "keywords",
                                    }
                                )
                            )
                        )
                        typer.echo(f"\nAnalysis:\n{analysis}")
                    else:
                        typer.echo(
                            f"Failed to parse: {result.get('error', 'Unknown error')}"
                        )
                else:
                    typer.echo(f"Analyzing text...")
                    analysis = asyncio.run(
                        analyst.run(
                            json.dumps(
                                {
                                    "text": target,
                                    "analysis_type": "summary",
                                }
                            )
                        )
                    )
                    typer.echo(f"\nAnalysis:\n{analysis}")

            elif command == "compare":
                raw = " ".join(args)
                if " vs " not in raw and " versus " not in raw:
                    typer.echo("Usage: /compare TOPIC1 vs TOPIC2")
                    continue
                topics = [
                    t.strip() for t in raw.replace(" versus ", " vs ").split(" vs ", 1)
                ]
                if len(topics) != 2:
                    typer.echo("Usage: /compare TOPIC1 vs TOPIC2")
                    continue
                typer.echo(f"Comparing: '{topics[0]}' vs '{topics[1]}'")
                results = []
                for topic in topics:
                    typer.echo(f"  Searching '{topic}'...")
                    r = asyncio.run(search.run(topic))
                    results.append({"topic": topic, "results": r})
                report = synthesizer.synthesize(
                    query=f"Comparison: {topics[0]} vs {topics[1]}",
                    findings=[
                        {
                            "title": f"[{r['topic']}]",
                            "content": r["results"][:1000],
                            "url": "",
                        }
                        for r in results
                    ],
                    format="markdown",
                )
                typer.echo(f"\n{report}")

            elif command == "turboquant":
                sub = " ".join(args).lower()
                if sub == "on":
                    config.turboquant.enabled = True
                    typer.echo("TurboQuant: enabled")
                elif sub == "off":
                    config.turboquant.enabled = False
                    typer.echo("TurboQuant: disabled")
                elif sub == "status":
                    tq = config.turboquant
                    typer.echo(f"TurboQuant: {'enabled' if tq.enabled else 'disabled'}")
                    typer.echo(f"  Method: {tq.method}")
                    typer.echo(f"  Target bits: {tq.target_bits}")
                    typer.echo(f"  Block size: {tq.block_size}")
                    typer.echo(f"  Fidelity target: {tq.fidelity_target}")
                else:
                    typer.echo("Usage: /turboquant [on|off|status]")

            elif command == "context":
                val = " ".join(args).lower()
                mapping = {"32k": 32768, "64k": 65536, "128k": 131072}
                if val in mapping:
                    context_window = mapping[val]
                    config.model.max_context = context_window
                    typer.echo(f"Context window: {val} ({context_window} tokens)")
                else:
                    typer.echo("Usage: /context [32k|64k|128k]")

            elif command == "model":
                name = " ".join(args)
                if not name:
                    typer.echo("Usage: /model NAME")
                    continue
                from ..core.models import registry

                existing = registry.get(name)
                if existing:
                    config.model.path = existing.path
                    typer.echo(f"Switched to: {name} ({existing.path})")
                else:
                    config.model.path = name
                    typer.echo(f"Model set to: {name}")

            elif command == "save":
                filename = (
                    " ".join(args)
                    or f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                path = Path(filename)
                path.write_text(json.dumps(chat_history, indent=2))
                typer.echo(f"Saved {len(chat_history)} messages to {path}")

            elif command == "history":
                if not chat_history:
                    typer.echo("No history yet.")
                else:
                    for i, msg in enumerate(
                        chat_history[-10:], max(1, len(chat_history) - 9)
                    ):
                        typer.echo(f"  [{i}] {msg['role']}: {msg['content'][:80]}...")

            elif command == "clear":
                chat_history.clear()
                typer.echo("Chat history cleared.")

            else:
                typer.echo(f"Unknown command: /{command}. Type /help for commands.")
        else:
            chat_history.append(
                {"role": "user", "content": user_input, "timestamp": time.time()}
            )
            typer.echo("Use /research, /search, /plan, /summarize, /analyze, or /help")


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
    import time

    from ..inference.mlx_backend import MLXBackend
    from ..turboquant.compressor import KVCacheCompressor

    model_path = model or "mlx-community/Llama-3.2-3B-Instruct-4bit"
    typer.echo(f"Benchmark: {model_path}")
    typer.echo(f"Prompt: {prompt_length} tokens, Generation: {gen_length} tokens")
    typer.echo(f"Runs: {runs} (warmup: {warmup})")
    typer.echo(f"TurboQuant: {'enabled' if turboquant else 'disabled'}")
    typer.echo("")

    backend = MLXBackend(
        model_path=model_path,
        turboquant=turboquant,
    )

    if not backend.load():
        typer.echo("Failed to load model. Install mlx-lm: pip install mlx-lm")
        raise typer.Exit(1)

    prompt = "The quick brown fox jumps over the lazy dog. " * (prompt_length // 9)

    typer.echo("Warming up...")
    for _ in range(warmup):
        backend.generate(prompt, max_tokens=32)

    typer.echo(f"Running {runs} benchmark iterations...")
    typer.echo("")

    prompt_times = []
    gen_times = []
    gen_tokens = []

    for i in range(runs):
        typer.echo(f"  Run {i + 1}/{runs}...", nl=False)
        result = backend.generate(prompt, max_tokens=gen_length)
        prompt_times.append(result.prompt_tokens / max(result.generation_time, 0.001))
        gen_times.append(result.tokens_generated / max(result.generation_time, 0.001))
        gen_tokens.append(result.tokens_generated)
        typer.echo(f" {result.tokens_per_second:.1f} tok/s")

    avg_prompt = sum(prompt_times) / len(prompt_times)
    avg_gen = sum(gen_times) / len(gen_times)
    avg_tokens = sum(gen_tokens) / len(gen_tokens)

    typer.echo("")
    typer.echo(f"{'─' * 50}")
    typer.echo(f"  Prompt throughput:  {avg_prompt:>8.1f} tok/s")
    typer.echo(f"  Gen throughput:    {avg_gen:>8.1f} tok/s")
    typer.echo(f"  Avg tokens:        {avg_tokens:>8.1f}")
    typer.echo(f"  Model:             {model_path}")
    typer.echo(f"  TurboQuant:        {'Yes' if turboquant else 'No'}")
    typer.echo(f"{'─' * 50}")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
