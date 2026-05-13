from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import httpx
from bs4 import BeautifulSoup

from .state import AgentState, TodoItem


MAX_FILE_BYTES = 400_000
MAX_OUTPUT_CHARS = 16_000
SHELL_TIMEOUT_SECS = 120
BG_LOG_TAIL_LINES = 200


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n... [truncated {len(text) - limit} chars]"


def _resolve(path: str, cwd: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = cwd / p
    return p.resolve()


# ---------- File tools ----------

def read_file(path: str, *, state: AgentState) -> str:
    p = _resolve(path, state.cwd)
    if not p.exists():
        return f"ERROR: {p} does not exist"
    if p.is_dir():
        return f"ERROR: {p} is a directory, use list_dir"
    data = p.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        return f"ERROR: file is {len(data)} bytes (limit {MAX_FILE_BYTES})"
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return f"ERROR: {p} is not valid UTF-8 (binary file)"


def write_file(path: str, content: str, *, state: AgentState) -> str:
    p = _resolve(path, state.cwd)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {p}"


def edit_file(path: str, old: str, new: str, *, state: AgentState) -> str:
    p = _resolve(path, state.cwd)
    if not p.exists():
        return f"ERROR: {p} does not exist"
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0:
        return f"ERROR: old string not found in {p}"
    if count > 1:
        return f"ERROR: old string appears {count} times in {p}; make it unique with more context"
    p.write_text(text.replace(old, new, 1), encoding="utf-8")
    return f"Replaced 1 occurrence in {p}"


def multi_edit(path: str, edits: list[dict[str, str]], *, state: AgentState) -> str:
    """Apply many (old, new) replacements to a single file atomically."""
    p = _resolve(path, state.cwd)
    if not p.exists():
        return f"ERROR: {p} does not exist"
    if not isinstance(edits, list) or not edits:
        return "ERROR: edits must be a non-empty list of {old, new} objects"
    text = p.read_text(encoding="utf-8")
    original = text
    for i, e in enumerate(edits, 1):
        if not isinstance(e, dict) or "old" not in e or "new" not in e:
            return f"ERROR: edit #{i} must have 'old' and 'new' fields"
        old, new = e["old"], e["new"]
        count = text.count(old)
        if count == 0:
            return f"ERROR: edit #{i}: old string not found in current file state"
        if count > 1:
            return f"ERROR: edit #{i}: old string appears {count} times; make it unique"
        text = text.replace(old, new, 1)
    if text == original:
        return f"No changes (all edits matched but produced identical text) in {p}"
    p.write_text(text, encoding="utf-8")
    return f"Applied {len(edits)} edits to {p}"


def list_dir(path: str = ".", *, state: AgentState) -> str:
    p = _resolve(path, state.cwd)
    if not p.exists():
        return f"ERROR: {p} does not exist"
    if not p.is_dir():
        return f"ERROR: {p} is not a directory"
    entries = []
    for child in sorted(p.iterdir()):
        kind = "DIR " if child.is_dir() else "FILE"
        try:
            size = child.stat().st_size if child.is_file() else 0
        except OSError:
            size = 0
        entries.append(f"{kind}  {size:>10}  {child.name}")
    return "\n".join(entries) if entries else "(empty)"


def search_files(pattern: str, path: str = ".", *, state: AgentState) -> str:
    p = _resolve(path, state.cwd)
    if not p.is_dir():
        return f"ERROR: {p} is not a directory"
    matches = list(p.rglob(pattern))[:500]
    if not matches:
        return f"No matches for {pattern} in {p}"
    return "\n".join(str(m) for m in matches)


def grep(pattern: str, path: str = ".", *, state: AgentState) -> str:
    p = _resolve(path, state.cwd)
    if not p.exists():
        return f"ERROR: {p} does not exist"
    rg_result = _grep_ripgrep(pattern, p)
    if rg_result is not None:
        return rg_result
    return _grep_python(pattern, p)


def _grep_ripgrep(pattern: str, p: Path) -> str | None:
    """Run ripgrep if it's on PATH. Returns None if rg is unavailable or rejects the pattern,
    in which case the caller falls back to the Python implementation."""
    rg = shutil.which("rg")
    if rg is None:
        return None
    try:
        proc = subprocess.run(
            [rg, "--no-heading", "--with-filename", "-n", "--color=never", pattern, str(p)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    # rg exit codes: 0 = matches, 1 = no matches, 2+ = error (often a regex feature
    # like lookaround that Python supports but the Rust regex crate does not). Fall back
    # on error so behavior stays a strict superset of the old Python-only implementation.
    if proc.returncode >= 2:
        return None
    if proc.returncode == 1 or not proc.stdout:
        return f"No matches for /{pattern}/"
    import re
    lines = proc.stdout.splitlines()
    truncated = len(lines) > 300
    if truncated:
        lines = lines[:300]
    # rg emits "path:line:content"; normalize to "path:line: content" to match the
    # documented format. Non-greedy `.+?` correctly handles Windows paths like C:\foo.py:42:.
    formatted = [re.sub(r"^(.+?:\d+:)", r"\1 ", line, count=1) for line in lines]
    if truncated:
        formatted.append("... [truncated to 300 matches]")
    return "\n".join(formatted)


def _grep_python(pattern: str, p: Path) -> str:
    import re
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"ERROR: bad regex: {e}"
    out: list[str] = []
    files: list[Path] = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
    for f in files[:5000]:
        try:
            with f.open("r", encoding="utf-8", errors="ignore") as fh:
                for i, line in enumerate(fh, 1):
                    if rx.search(line):
                        out.append(f"{f}:{i}: {line.rstrip()}")
                        if len(out) >= 300:
                            out.append("... [truncated to 300 matches]")
                            return "\n".join(out)
        except (OSError, UnicodeDecodeError):
            continue
    return "\n".join(out) if out else f"No matches for /{pattern}/"


# ---------- Shell ----------

def _shell_args(command: str) -> list[str]:
    if platform.system() == "Windows":
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
    return ["bash", "-lc", command]


def run_shell(command: str, *, state: AgentState) -> str:
    """Run a foreground command. Will time out at SHELL_TIMEOUT_SECS. Do NOT use for servers."""
    try:
        proc = subprocess.run(
            _shell_args(command),
            cwd=str(state.cwd),
            capture_output=True,
            text=True,
            timeout=SHELL_TIMEOUT_SECS,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return (
            f"ERROR: command timed out after {SHELL_TIMEOUT_SECS}s. "
            "If this is a long-running server, use run_shell_background instead."
        )
    except FileNotFoundError as e:
        return f"ERROR: shell not found: {e}"
    parts = []
    if proc.stdout:
        parts.append("STDOUT:\n" + proc.stdout)
    if proc.stderr:
        parts.append("STDERR:\n" + proc.stderr)
    parts.append(f"EXIT: {proc.returncode}")
    return _truncate("\n".join(parts))


def run_shell_background(command: str, *, state: AgentState) -> str:
    """Spawn a process detached from the agent. Returns immediately with a PID."""
    log_dir = state.cwd / ".aquila"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"bg_{abs(hash(command)) % 10_000_000}.log"
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    creationflags = 0
    if platform.system() == "Windows":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    try:
        proc = subprocess.Popen(
            _shell_args(command),
            cwd=str(state.cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
    except Exception as e:
        log.close()
        return f"ERROR: failed to start background process: {e}"
    state.bg_processes[proc.pid] = proc
    return (
        f"Started PID {proc.pid}: {command}\n"
        f"Logs: {log_path}\n"
        f"Use list_processes / stop_process / read_file on the log path to inspect."
    )


def list_processes(*, state: AgentState) -> str:
    if not state.bg_processes:
        return "(no background processes running)"
    lines = ["PID     STATUS    COMMAND"]
    dead: list[int] = []
    for pid, p in list(state.bg_processes.items()):
        rc = p.poll()
        if rc is None:
            status = "running"
        else:
            status = f"exited({rc})"
            dead.append(pid)
        cmd = " ".join(p.args) if isinstance(p.args, list) else str(p.args)
        lines.append(f"{pid:<7} {status:<9} {cmd[:120]}")
    for pid in dead:
        state.bg_processes.pop(pid, None)
    return "\n".join(lines)


def stop_process(pid: int, *, state: AgentState) -> str:
    proc = state.bg_processes.get(pid)
    if proc is None:
        return f"ERROR: no background process with PID {pid}"
    if proc.poll() is not None:
        state.bg_processes.pop(pid, None)
        return f"PID {pid} already exited"
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception as e:
        return f"ERROR: failed to stop {pid}: {e}"
    state.bg_processes.pop(pid, None)
    return f"Stopped PID {pid}"


# ---------- Web ----------

def web_search(query: str, max_results: int = 5, *, state: AgentState) -> str:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "ERROR: duckduckgo-search not installed"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"ERROR: search failed: {e}"
    if not results:
        return f"No results for: {query}"
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("href") or r.get("url", "")
        body = r.get("body", "")
        lines.append(f"{i}. {title}\n   {url}\n   {body}")
    return "\n\n".join(lines)


def web_fetch(url: str, *, state: AgentState) -> str:
    try:
        with httpx.Client(follow_redirects=True, timeout=30, headers={"User-Agent": "aquila/0.1"}) as c:
            r = c.get(url)
            r.raise_for_status()
    except Exception as e:
        return f"ERROR: fetch failed: {e}"
    ct = r.headers.get("content-type", "")
    if "html" in ct:
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
        return _truncate(text)
    return _truncate(r.text)


# ---------- Todo ----------

VALID_TODO_STATUSES = {"pending", "in_progress", "completed"}


def todo_write(todos: list[dict[str, str]], *, state: AgentState) -> str:
    """Replace the active todo list. Each item needs content, activeForm, status."""
    if not isinstance(todos, list):
        return "ERROR: todos must be a list"
    parsed: list[TodoItem] = []
    for i, t in enumerate(todos, 1):
        if not isinstance(t, dict):
            return f"ERROR: todo #{i} must be an object"
        for key in ("content", "activeForm", "status"):
            if key not in t:
                return f"ERROR: todo #{i} missing '{key}'"
        if t["status"] not in VALID_TODO_STATUSES:
            return f"ERROR: todo #{i} status must be one of {sorted(VALID_TODO_STATUSES)}"
        parsed.append(TodoItem(
            content=str(t["content"]),
            activeForm=str(t["activeForm"]),
            status=str(t["status"]),
        ))
    in_progress = [t for t in parsed if t.status == "in_progress"]
    if len(in_progress) > 1:
        return "ERROR: only one todo may be in_progress at a time"
    state.todos = parsed
    done = sum(1 for t in parsed if t.status == "completed")
    return f"Todo list updated ({done}/{len(parsed)} complete)"


# ---------- Tool registry / schemas ----------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full UTF-8 text contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative or absolute path"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the given content. Creates parent dirs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exactly one occurrence of `old` with `new` in a file. `old` must be unique within the file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multi_edit",
            "description": (
                "Apply multiple (old, new) replacements to a single file in one call. "
                "Edits are applied in order; each `old` must be unique in the file *at the moment* the edit is applied. "
                "All-or-nothing: if any edit fails the file is not modified."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"old": {"type": "string"}, "new": {"type": "string"}},
                            "required": ["old", "new"],
                        },
                    },
                },
                "required": ["path", "edits"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List entries of a directory (non-recursive).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Recursive glob search. Pattern like '**/*.py'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Recursive regex search across text files. Returns 'file:line: content' matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Execute a SHORT foreground shell command (PowerShell on Windows, bash on Unix). "
                "Returns stdout/stderr/exit code. Times out after 120s. "
                "DO NOT use for dev servers, watchers, REPLs, or anything that does not exit on its own — use run_shell_background for those."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell_background",
            "description": (
                "Start a LONG-RUNNING command (dev server, watcher, etc.) in the background. "
                "Returns immediately with a PID. Output goes to a log file at .aquila/bg_*.log inside the cwd. "
                "Use this for `npm run dev`, `npx serve`, `vite`, `python -m http.server`, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_processes",
            "description": "List background processes started via run_shell_background, with status.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_process",
            "description": "Terminate a background process by PID.",
            "parameters": {
                "type": "object",
                "properties": {"pid": {"type": "integer"}},
                "required": ["pid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web via DuckDuckGo. Returns title/url/snippet for top results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a URL and return the page as plain text (HTML stripped).",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": (
                "Maintain a structured task list for the current request. "
                "Call this at the start of any multi-step task to outline the plan, then call it again to update statuses as you progress. "
                "Each call REPLACES the entire list. Exactly one item may have status 'in_progress' at a time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "Imperative form, e.g. 'Create index.html'"},
                                "activeForm": {"type": "string", "description": "Present continuous, e.g. 'Creating index.html'"},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                            },
                            "required": ["content", "activeForm", "status"],
                        },
                    },
                },
                "required": ["todos"],
            },
        },
    },
]


READONLY_TOOL_NAMES = {
    "read_file", "list_dir", "search_files", "grep",
    "web_search", "web_fetch", "list_processes", "todo_write",
}


def build_dispatcher(state: AgentState) -> dict[str, Callable[..., str]]:
    return {
        "read_file": lambda path: read_file(path, state=state),
        "write_file": lambda path, content: write_file(path, content, state=state),
        "edit_file": lambda path, old, new: edit_file(path, old, new, state=state),
        "multi_edit": lambda path, edits: multi_edit(path, edits, state=state),
        "list_dir": lambda path=".": list_dir(path, state=state),
        "search_files": lambda pattern, path=".": search_files(pattern, path, state=state),
        "grep": lambda pattern, path=".": grep(pattern, path, state=state),
        "run_shell": lambda command: run_shell(command, state=state),
        "run_shell_background": lambda command: run_shell_background(command, state=state),
        "list_processes": lambda: list_processes(state=state),
        "stop_process": lambda pid: stop_process(int(pid), state=state),
        "web_search": lambda query, max_results=5: web_search(query, max_results, state=state),
        "web_fetch": lambda url: web_fetch(url, state=state),
        "todo_write": lambda todos: todo_write(todos, state=state),
    }


def execute_tool(name: str, raw_args: str, dispatcher: dict[str, Callable[..., str]]) -> str:
    fn = dispatcher.get(name)
    if fn is None:
        return f"ERROR: unknown tool '{name}'"
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError as e:
        return f"ERROR: invalid JSON arguments: {e}"
    if not isinstance(args, dict):
        return "ERROR: tool arguments must be a JSON object"
    try:
        result = fn(**args)
    except TypeError as e:
        return f"ERROR: bad arguments for {name}: {e}"
    except Exception as e:
        return f"ERROR: {name} raised: {e}"
    return _truncate(str(result))
