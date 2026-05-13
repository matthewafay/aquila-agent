from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .client import LMStudioClient
from .state import AgentState
from .tools import (
    READONLY_TOOL_NAMES,
    TOOL_SCHEMAS,
    build_dispatcher,
    execute_tool,
)


SYSTEM_PROMPT = """You are aquila, a terminal coding agent running locally against an LM Studio model.

You have tools to read/write/edit files, list directories, run shell commands (foreground and background), glob, grep, search the web, fetch URLs, and maintain a structured task list. You operate the user's machine directly. Your job is to DO the work yourself by calling tools, not to instruct the user on how to do it.

ABSOLUTE RULES:
- When the user asks you to build, create, scaffold, fix, refactor, or implement something, you MUST produce that work by calling write_file / edit_file / multi_edit / run_shell. Do NOT reply with a numbered tutorial of steps for the user to perform.
- An empty working directory is not a blocker. It is a green light to start creating files from scratch.
- After every tool call, KEEP GOING until the task is finished. Stop only when the deliverable exists on disk and works, or when you genuinely need information only the user has.
- Never tell the user to run `npm install`, `git init`, etc. — call run_shell and run it yourself.
- For long-running commands (dev servers, watchers, `npx serve`, `npm run dev`, `python -m http.server`, `vite`, etc.) you MUST use run_shell_background, NEVER run_shell. run_shell will block and time out on servers.

Task tracking (IMPORTANT):
- For any non-trivial task (3+ steps), CALL todo_write at the start to lay out the plan as a list of items with status "pending", with one starting as "in_progress".
- As you finish each step, CALL todo_write again with the updated list — flip the completed item to "completed" and the next item to "in_progress".
- Keep todos crisp and specific. Use the imperative form ("Create index.html") in `content` and the present continuous ("Creating index.html") in `activeForm`.

Editing files:
- Prefer edit_file or multi_edit over write_file when modifying an existing file.
- Use multi_edit when you need to make several changes to one file in a single round-trip.

Style:
- Be concise. No filler. No emojis unless asked.
- For Windows shell commands use PowerShell syntax; on Unix use bash.
- When finished, give a one or two sentence summary of what was created/changed and how the user runs it.

The user's working directory and platform are below. All relative paths resolve against the working directory."""


PLAN_MODE_ADDENDUM = """

PLAN MODE IS ACTIVE.

You may ONLY use read-only tools: read_file, list_dir, search_files, grep, web_search, web_fetch, list_processes, todo_write.
You MUST NOT call write_file, edit_file, multi_edit, run_shell, run_shell_background, or stop_process. They are unavailable.

Your job in plan mode:
1. Investigate the task using the read-only tools above.
2. Use todo_write to capture the steps you would take.
3. Produce a numbered, step-by-step plan describing exactly what changes you will make once the user approves. For each step say which file(s) it touches and what the change does.
4. End with a one-line summary like "Approve with /approve, or /cancel to abort."

Do not perform the work yet. The actual edits happen after approval."""


@dataclass
class AgentConfig:
    model: str
    cwd: Path
    max_tool_iters: int = 25
    temperature: float = 0.2
    stream: bool = True


@dataclass
class Agent:
    client: LMStudioClient
    config: AgentConfig
    messages: list[dict[str, Any]] = field(default_factory=list)
    on_event: Callable[[str, dict[str, Any]], None] | None = None
    plan_mode: bool = False

    def __post_init__(self) -> None:
        self.state = AgentState(cwd=self.config.cwd)
        self._dispatcher = build_dispatcher(self.state)
        if not self.messages:
            self.messages.append({
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    + f"\n\nWorking directory: {self.state.cwd}"
                    + f"\nPlatform: {platform.system()} {platform.release()}"
                ),
            })

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.on_event:
            self.on_event(kind, data)

    def reset(self) -> None:
        system = self.messages[0] if self.messages and self.messages[0]["role"] == "system" else None
        self.messages = [system] if system else []
        self.plan_mode = False
        self.state.todos = []

    def set_model(self, model: str) -> None:
        self.config.model = model

    def set_cwd(self, cwd: Path) -> None:
        self.state.cwd = cwd
        self.config.cwd = cwd

    def enter_plan_mode(self) -> None:
        if self.plan_mode:
            return
        self.plan_mode = True
        self.messages.append({"role": "system", "content": PLAN_MODE_ADDENDUM})

    def exit_plan_mode(self) -> None:
        if not self.plan_mode:
            return
        self.plan_mode = False
        self.messages.append({
            "role": "system",
            "content": "Plan mode ended. All tools are available again.",
        })

    def _active_tools(self) -> list[dict[str, Any]]:
        if not self.plan_mode:
            return TOOL_SCHEMAS
        return [t for t in TOOL_SCHEMAS if t["function"]["name"] in READONLY_TOOL_NAMES]

    # ---------- main turn ----------

    def send(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        for iteration in range(1, self.config.max_tool_iters + 1):
            try:
                content, tool_calls = self._one_round(iteration)
            except Exception as e:
                self._emit("error", {"message": f"Model call failed: {e}"})
                self.messages.pop()  # let the user retry
                return f"[error] {e}"

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"] or "{}"},
                    }
                    for tc in tool_calls
                ]
            self.messages.append(assistant_msg)

            if not tool_calls:
                return content or ""

            for tc in tool_calls:
                name = tc["name"]
                raw_args = tc["arguments"] or "{}"
                self._emit("tool_call", {"name": name, "arguments": raw_args})
                if self.plan_mode and name not in READONLY_TOOL_NAMES:
                    result = (
                        f"ERROR: '{name}' is blocked in plan mode. "
                        "Only read-only tools are allowed; produce a plan instead."
                    )
                else:
                    result = execute_tool(name, raw_args, self._dispatcher)
                self._emit("tool_result", {"name": name, "result": result, "arguments": raw_args})
                if name == "todo_write":
                    self._emit("todos", {"todos": [
                        {"content": t.content, "activeForm": t.activeForm, "status": t.status}
                        for t in self.state.todos
                    ]})
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

        self._emit("error", {"message": "Hit max tool iterations"})
        return "[stopped: max tool iterations reached]"

    # ---------- single model round-trip ----------

    def _one_round(self, iteration: int) -> tuple[str, list[dict[str, Any]]]:
        """Returns (assistant_text, tool_call_dicts)."""
        if self.config.stream:
            return self._stream_round(iteration)
        return self._nonstream_round(iteration)

    def _nonstream_round(self, iteration: int) -> tuple[str, list[dict[str, Any]]]:
        self._emit("turn_start", {"iteration": iteration, "model": self.config.model})
        resp = self.client.chat(
            model=self.config.model,
            messages=self.messages,
            tools=self._active_tools(),
            temperature=self.config.temperature,
            stream=False,
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        tcs = []
        for tc in getattr(msg, "tool_calls", None) or []:
            tcs.append({"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments or "{}"})
        self._emit("turn_end", {"iteration": iteration})
        if content:
            self._emit("assistant_text", {"text": content})
        return content, tcs

    def _stream_round(self, iteration: int) -> tuple[str, list[dict[str, Any]]]:
        self._emit("turn_start", {"iteration": iteration, "model": self.config.model})
        start = time.monotonic()
        content_parts: list[str] = []
        # tool calls arrive as fragments keyed by index
        tc_acc: dict[int, dict[str, Any]] = {}
        token_count = 0

        stream = self.client.chat(
            model=self.config.model,
            messages=self.messages,
            tools=self._active_tools(),
            temperature=self.config.temperature,
            stream=True,
        )
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
                token_count += 1
                self._emit("stream_delta", {
                    "text": delta.content,
                    "elapsed": time.monotonic() - start,
                    "tokens": token_count,
                })
            for tc_chunk in getattr(delta, "tool_calls", None) or []:
                idx = tc_chunk.index
                slot = tc_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if getattr(tc_chunk, "id", None):
                    slot["id"] = tc_chunk.id
                fn = getattr(tc_chunk, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["arguments"] += fn.arguments
                token_count += 1
                self._emit("stream_progress", {
                    "elapsed": time.monotonic() - start,
                    "tokens": token_count,
                })
        self._emit("turn_end", {"iteration": iteration, "elapsed": time.monotonic() - start, "tokens": token_count})

        content = "".join(content_parts)
        if content:
            self._emit("assistant_text", {"text": content})
        tcs = [tc_acc[i] for i in sorted(tc_acc.keys())]
        return content, tcs
