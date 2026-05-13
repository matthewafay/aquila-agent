from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .agent import Agent
from .client import LMStudioClient


SLASH_COMMANDS = [
    "/help", "/model", "/models", "/clear", "/cwd", "/history",
    "/plan", "/approve", "/exec", "/cancel",
    "/todos", "/processes", "/stop",
    "/exit", "/quit",
]


EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".jsx": "jsx",
    ".html": "html", ".htm": "html", ".css": "css", ".scss": "scss",
    ".json": "json", ".jsonc": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".md": "markdown", ".sh": "bash", ".ps1": "powershell",
    ".sql": "sql", ".rs": "rust", ".go": "go", ".java": "java",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
    ".rb": "ruby", ".php": "php", ".swift": "swift", ".kt": "kotlin",
    ".xml": "xml", ".lua": "lua",
}


def _lang_from_path(path: str) -> str:
    ext = Path(path).suffix.lower()
    return EXT_TO_LANG.get(ext, "text")


def _render_todos(todos: list[dict[str, Any]]) -> Panel:
    if not todos:
        return Panel("(no todos)", title="todos", border_style="dim")
    lines = []
    for t in todos:
        status = t.get("status", "pending")
        if status == "completed":
            mark = "[green]✓[/]"
            style = "dim strike"
        elif status == "in_progress":
            mark = "[yellow]▶[/]"
            style = "bold yellow"
        else:
            mark = "[dim]○[/]"
            style = "white"
        text = t.get("activeForm") if status == "in_progress" else t.get("content", "")
        lines.append(f"{mark} [{style}]{text}[/]")
    done = sum(1 for t in todos if t.get("status") == "completed")
    title = f"todos  {done}/{len(todos)}"
    return Panel("\n".join(lines), title=title, border_style="blue", expand=False)


class EventRenderer:
    """Renders agent lifecycle events to a Rich console with live streaming."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._live: Live | None = None
        self._stream_buf: list[str] = []
        self._status_iter = 0
        self._status_elapsed = 0.0
        self._status_tokens = 0
        self._todos: list[dict[str, Any]] = []

    # ---------- public dispatch ----------

    def __call__(self, kind: str, data: dict[str, Any]) -> None:
        handler = getattr(self, f"_on_{kind}", None)
        if handler is None:
            return
        handler(data)

    # ---------- streaming ----------

    def _status_text(self) -> Text:
        t = Text()
        t.append("◐ ", style="cyan")
        t.append(f"iter {self._status_iter}", style="dim")
        t.append(f"  ·  {self._status_elapsed:5.1f}s", style="dim")
        t.append(f"  ·  {self._status_tokens} tok", style="dim")
        return t

    def _live_renderable(self) -> Group:
        text = "".join(self._stream_buf).strip()
        if text:
            body: Any = Markdown(text)
        else:
            body = Text("…", style="dim italic")
        return Group(body, self._status_text())

    def _ensure_live(self) -> Live:
        if self._live is None:
            self._stream_buf = []
            self._live = Live(
                self._live_renderable(),
                console=self.console,
                refresh_per_second=12,
                transient=False,
            )
            self._live.__enter__()
        return self._live

    def _end_live(self, *, keep_text: bool) -> None:
        if self._live is None:
            return
        if keep_text:
            text = "".join(self._stream_buf).strip()
            self._live.update(Markdown(text) if text else Text(""))
        else:
            # erase status line by replacing with empty
            self._live.update(Text(""))
        self._live.__exit__(None, None, None)
        self._live = None
        self._stream_buf = []

    # ---------- event handlers ----------

    def _on_turn_start(self, data: dict[str, Any]) -> None:
        self._status_iter = int(data.get("iteration", 0))
        self._status_elapsed = 0.0
        self._status_tokens = 0
        self._stream_buf = []
        self._ensure_live()

    def _on_stream_delta(self, data: dict[str, Any]) -> None:
        self._stream_buf.append(data.get("text", ""))
        self._status_elapsed = float(data.get("elapsed", self._status_elapsed))
        self._status_tokens = int(data.get("tokens", self._status_tokens))
        if self._live is not None:
            self._live.update(self._live_renderable())

    def _on_stream_progress(self, data: dict[str, Any]) -> None:
        self._status_elapsed = float(data.get("elapsed", self._status_elapsed))
        self._status_tokens = int(data.get("tokens", self._status_tokens))
        if self._live is not None:
            self._live.update(self._live_renderable())

    def _on_turn_end(self, data: dict[str, Any]) -> None:
        # if there was streamed text, keep it printed; otherwise wipe spinner
        keep = bool("".join(self._stream_buf).strip())
        self._end_live(keep_text=keep)

    def _on_assistant_text(self, data: dict[str, Any]) -> None:
        # In streaming mode the text is already shown; only print here if we
        # didn't stream (non-streaming fallback).
        if self._live is not None:
            return
        text = (data.get("text") or "").strip()
        if text:
            self.console.print(Markdown(text))

    def _on_tool_call(self, data: dict[str, Any]) -> None:
        name = data["name"]
        args = data.get("arguments", "{}")
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            parsed = None

        # special: write_file → show content as syntax-highlighted code
        if name == "write_file" and isinstance(parsed, dict) and "content" in parsed:
            path = parsed.get("path", "")
            content = parsed.get("content", "")
            preview = content if len(content) < 4000 else content[:4000] + f"\n... [+{len(content) - 4000} chars]"
            self.console.print(Panel(
                Syntax(preview, _lang_from_path(path), theme="ansi_dark", line_numbers=True, word_wrap=False),
                title=f"[bold cyan]write_file[/]  {path}  [dim]({len(content)} chars)[/]",
                border_style="cyan",
                expand=False,
            ))
            return

        # special: edit_file → show diff
        if name == "edit_file" and isinstance(parsed, dict) and "old" in parsed and "new" in parsed:
            self._render_edit(parsed.get("path", ""), parsed["old"], parsed["new"], multi=False)
            return

        # special: multi_edit → show each edit as a mini diff
        if name == "multi_edit" and isinstance(parsed, dict) and isinstance(parsed.get("edits"), list):
            path = parsed.get("path", "")
            edits = parsed["edits"]
            self.console.print(Panel(
                Text.from_markup(f"[dim]applying {len(edits)} edits to[/] {path}"),
                title="[bold cyan]multi_edit[/]",
                border_style="cyan",
                expand=False,
            ))
            for i, e in enumerate(edits[:8], 1):
                self._render_edit(path, e.get("old", ""), e.get("new", ""), multi=True, label=f"edit #{i}")
            if len(edits) > 8:
                self.console.print(Text(f"  … +{len(edits) - 8} more edits", style="dim"))
            return

        # default tool-call rendering
        try:
            pretty = json.dumps(parsed if parsed is not None else json.loads(args), indent=2)
        except (json.JSONDecodeError, TypeError):
            pretty = args
        self.console.print(Panel(
            Syntax(pretty, "json", theme="ansi_dark", word_wrap=True),
            title=f"[bold cyan]tool[/] {name}",
            border_style="cyan",
            expand=False,
        ))

    def _render_edit(self, path: str, old: str, new: str, *, multi: bool, label: str | None = None) -> None:
        diff_lines = list(difflib.unified_diff(
            old.splitlines(keepends=False),
            new.splitlines(keepends=False),
            fromfile="before",
            tofile="after",
            lineterm="",
            n=2,
        ))
        body = "\n".join(diff_lines) if diff_lines else "(no textual change)"
        title = label or f"edit_file  {path}"
        self.console.print(Panel(
            Syntax(body, "diff", theme="ansi_dark", word_wrap=False),
            title=f"[bold magenta]{title}[/]" + (f"  [dim]{path}[/]" if multi else ""),
            border_style="magenta",
            expand=False,
        ))

    def _on_tool_result(self, data: dict[str, Any]) -> None:
        name = data["name"]
        result = data.get("result", "")
        # condense write_file / edit_file results — they're just "Wrote N chars"
        if name in ("write_file", "edit_file", "multi_edit"):
            self.console.print(Text(f"  → {result}", style="green dim"))
            return
        preview = result if len(result) < 1500 else result[:1500] + f"\n... [+{len(result) - 1500} chars]"
        self.console.print(Panel(
            preview,
            title=f"[bold green]result[/] {name}",
            border_style="green",
            expand=False,
        ))

    def _on_todos(self, data: dict[str, Any]) -> None:
        self._todos = data.get("todos", [])
        self.console.print(_render_todos(self._todos))

    def _on_error(self, data: dict[str, Any]) -> None:
        self.console.print(f"[bold red]error:[/] {data.get('message', '')}")


class REPL:
    def __init__(self, client: LMStudioClient, agent: Agent) -> None:
        self.client = client
        self.agent = agent
        self.console = Console()
        self.renderer = EventRenderer(self.console)
        history_path = Path.home() / ".lmcc_history"
        self.session: PromptSession[str] = PromptSession(
            history=FileHistory(str(history_path)),
            completer=WordCompleter(SLASH_COMMANDS, ignore_case=True, sentence=True),
        )
        self.agent.on_event = self.renderer

    # ---------- slash commands ----------

    def _cmd_help(self) -> None:
        table = Table(title="Commands", show_header=True, header_style="bold")
        table.add_column("command")
        table.add_column("description")
        rows = [
            ("/help", "show this help"),
            ("/models", "list models currently available in LM Studio"),
            ("/model <id>", "switch to a different model (substring match works)"),
            ("/clear", "wipe conversation history (keeps system prompt)"),
            ("/cwd [path]", "show or change the working directory"),
            ("/history", "print the raw message history"),
            ("/plan [task]", "enter plan mode (read-only tools); optional task to start"),
            ("/approve, /exec", "approve the plan and execute it with all tools"),
            ("/cancel", "leave plan mode without executing"),
            ("/todos", "show the current task list"),
            ("/processes", "show background processes"),
            ("/stop <pid>", "stop a background process by PID"),
            ("/exit, /quit", "leave the app (Ctrl+D also works)"),
        ]
        for k, v in rows:
            table.add_row(k, v)
        self.console.print(table)
        self.console.print(Text("Anything not starting with '/' is sent to the model.", style="dim"))

    def _cmd_models(self) -> None:
        try:
            models = self.client.list_models()
        except RuntimeError as e:
            self.console.print(f"[red]{e}[/]")
            return
        table = Table(title="LM Studio models", show_header=True, header_style="bold")
        table.add_column("id")
        table.add_column("active", justify="center")
        for m in models:
            active = "*" if m.id == self.agent.config.model else ""
            table.add_row(m.id, active)
        self.console.print(table)

    def _cmd_model(self, arg: str) -> None:
        target = arg.strip()
        if not target:
            self.console.print(f"current model: [bold]{self.agent.config.model}[/]")
            return
        try:
            ids = [m.id for m in self.client.list_models()]
        except RuntimeError as e:
            self.console.print(f"[red]{e}[/]")
            return
        if target not in ids:
            matches = [i for i in ids if target.lower() in i.lower()]
            if len(matches) == 1:
                target = matches[0]
            elif len(matches) > 1:
                self.console.print(f"[yellow]ambiguous, matches:[/] {', '.join(matches)}")
                return
            else:
                self.console.print(f"[red]no such model:[/] {target}")
                return
        self.agent.set_model(target)
        self.console.print(f"switched to [bold]{target}[/]")

    def _cmd_clear(self) -> None:
        self.agent.reset()
        self.console.clear()
        self.console.print("[dim]history cleared[/]")

    def _cmd_cwd(self, arg: str) -> None:
        if not arg.strip():
            self.console.print(str(self.agent.config.cwd))
            return
        new = Path(arg.strip()).expanduser().resolve()
        if not new.is_dir():
            self.console.print(f"[red]not a directory:[/] {new}")
            return
        self.agent.set_cwd(new)
        self.console.print(f"cwd: {new}")

    def _cmd_history(self) -> None:
        for m in self.agent.messages:
            role = m.get("role", "?")
            content = m.get("content", "") or ""
            if isinstance(content, list):
                content = json.dumps(content)
            self.console.print(Panel(content[:1000], title=role, border_style="dim"))

    def _cmd_plan(self, arg: str) -> None:
        if self.agent.plan_mode:
            self.console.print("[dim]already in plan mode[/]")
        else:
            self.agent.enter_plan_mode()
            self.console.print(Panel.fit(
                Text.from_markup(
                    "[bold yellow]plan mode[/] — only read-only tools are available.\n"
                    "Describe what you want; the model will investigate and propose a plan.\n"
                    "[dim]/approve to execute · /cancel to abandon[/]"
                ),
                border_style="yellow",
            ))
        if arg.strip():
            self.agent.send(arg)

    def _cmd_approve(self) -> None:
        if not self.agent.plan_mode:
            self.console.print("[yellow]not in plan mode[/]")
            return
        self.agent.exit_plan_mode()
        self.console.print("[green]plan approved — executing[/]")
        self.agent.send(
            "Execute the plan you just proposed using all available tools "
            "(write_file, edit_file, multi_edit, run_shell, run_shell_background). Report what you did when finished."
        )

    def _cmd_cancel(self) -> None:
        if not self.agent.plan_mode:
            self.console.print("[yellow]not in plan mode[/]")
            return
        self.agent.exit_plan_mode()
        self.console.print("[dim]plan cancelled[/]")

    def _cmd_todos(self) -> None:
        todos = [
            {"content": t.content, "activeForm": t.activeForm, "status": t.status}
            for t in self.agent.state.todos
        ]
        self.console.print(_render_todos(todos))

    def _cmd_processes(self) -> None:
        from .tools import list_processes
        self.console.print(list_processes(state=self.agent.state))

    def _cmd_stop(self, arg: str) -> None:
        from .tools import stop_process
        try:
            pid = int(arg.strip())
        except ValueError:
            self.console.print("[red]usage: /stop <pid>[/]")
            return
        self.console.print(stop_process(pid, state=self.agent.state))

    def _handle_slash(self, line: str) -> bool:
        parts = line.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        if cmd in ("/exit", "/quit"):
            return False
        if cmd == "/help":
            self._cmd_help()
        elif cmd == "/models":
            self._cmd_models()
        elif cmd == "/model":
            self._cmd_model(arg)
        elif cmd == "/clear":
            self._cmd_clear()
        elif cmd == "/cwd":
            self._cmd_cwd(arg)
        elif cmd == "/history":
            self._cmd_history()
        elif cmd == "/plan":
            self._cmd_plan(arg)
        elif cmd in ("/approve", "/exec"):
            self._cmd_approve()
        elif cmd == "/cancel":
            self._cmd_cancel()
        elif cmd == "/todos":
            self._cmd_todos()
        elif cmd == "/processes":
            self._cmd_processes()
        elif cmd == "/stop":
            self._cmd_stop(arg)
        else:
            self.console.print(f"[yellow]unknown command:[/] {cmd}  (try /help)")
        return True

    # ---------- main loop ----------

    def banner(self) -> None:
        self.console.print(Panel.fit(
            Text.from_markup(
                "[bold]lmcc[/] — local coding agent\n"
                f"model: [cyan]{self.agent.config.model}[/]\n"
                f"cwd:   [cyan]{self.agent.config.cwd}[/]\n"
                "[dim]/help for commands, Ctrl+D to exit[/]"
            ),
            border_style="magenta",
        ))

    def run(self) -> None:
        self.banner()
        try:
            while True:
                prompt_str = "plan» " if self.agent.plan_mode else "» "
                try:
                    line = self.session.prompt(prompt_str)
                except (EOFError, KeyboardInterrupt):
                    self.console.print()
                    return
                if not line.strip():
                    continue
                if line.lstrip().startswith("/"):
                    if not self._handle_slash(line):
                        return
                    continue
                try:
                    self.agent.send(line)
                except KeyboardInterrupt:
                    self.console.print("[yellow]interrupted[/]")
                except Exception as e:
                    self.console.print(f"[red]error:[/] {e}")
        finally:
            self.agent.state.cleanup()
