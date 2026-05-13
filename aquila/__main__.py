from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _force_utf8_io() -> None:
    """Make Rich's unicode glyphs survive on legacy Windows consoles / piped output."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


_force_utf8_io()

from rich.console import Console

from .agent import Agent, AgentConfig
from .client import DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_REQUEST_TIMEOUT, LMStudioClient
from .ui import REPL, EventRenderer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="aquila", description="Local coding agent backed by LM Studio.")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="LM Studio base URL (default %(default)s)")
    p.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key for the local server (default %(default)s)")
    p.add_argument("--model", default=None, help="Model id to use (defaults to the first loaded model)")
    p.add_argument("--cwd", default=None, help="Working directory for file/shell tools (defaults to current dir)")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-iters", type=int, default=25, help="Max tool-call iterations per user turn")
    p.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help=(
            "Per-request timeout in seconds for model calls to LM Studio "
            "(default %(default)s; also settable via AQUILA_REQUEST_TIMEOUT). "
            "Raise this if you hit 'Model call failed: timed out' on long generations."
        ),
    )
    p.add_argument(
        "--auto-compact-threshold",
        type=float,
        default=0.8,
        help=(
            "Fraction of the model's context window (0.0-1.0) at which "
            "older turns are auto-summarized to free space (default %(default)s). "
            "Set to 0 to disable auto-compaction; /compact remains available."
        ),
    )
    p.add_argument("prompt", nargs="*", help="If provided, run a one-shot prompt and exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()

    client = LMStudioClient(
        base_url=args.base_url,
        api_key=args.api_key,
        request_timeout=args.request_timeout,
    )

    try:
        models = client.list_models()
    except RuntimeError as e:
        console.print(f"[red]{e}[/]")
        console.print(
            "[dim]Start LM Studio, open the Developer tab, click 'Start Server', and load a model.[/]"
        )
        return 2

    if not models:
        console.print("[red]LM Studio responded but no models are loaded.[/]")
        console.print("[dim]Load a model in LM Studio (Developer tab) and try again.[/]")
        return 2

    model_id = args.model or models[0].id
    if args.model and args.model not in [m.id for m in models]:
        matches = [m.id for m in models if args.model.lower() in m.id.lower()]
        if len(matches) == 1:
            model_id = matches[0]
        else:
            console.print(f"[red]model '{args.model}' not found; loaded models:[/] {', '.join(m.id for m in models)}")
            return 2

    cwd = Path(args.cwd).expanduser().resolve() if args.cwd else Path.cwd()
    if not cwd.is_dir():
        console.print(f"[red]cwd is not a directory:[/] {cwd}")
        return 2

    cfg = AgentConfig(
        model=model_id,
        cwd=cwd,
        max_tool_iters=args.max_iters,
        temperature=args.temperature,
        auto_compact_threshold=max(0.0, min(1.0, args.auto_compact_threshold)),
    )
    agent = Agent(client=client, config=cfg)

    if args.prompt:
        prompt = " ".join(args.prompt)
        agent.on_event = EventRenderer()
        agent.send(prompt)
        return 0

    REPL(client, agent).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
