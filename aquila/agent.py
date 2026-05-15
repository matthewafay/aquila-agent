from __future__ import annotations

import json
import platform
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .client import LMStudioClient
from .images import build_user_message, find_images
from .state import AgentState
from .tools import (
    READONLY_TOOL_NAMES,
    TOOL_SCHEMAS,
    build_dispatcher,
    execute_tool,
    verify_touched_files,
)


MAX_VERIFICATION_CYCLES = 2
MAX_EMPTY_RETRIES = 2
MODEL_RETRY_DELAY_SECS = 2.0

# Strips <think>…</think> blocks that reasoning models (e.g. Qwen3, DeepSeek-R1)
# sometimes emit as plain content rather than in a separate field.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


try:
    # openai SDK exposes typed exception classes for transient failures.
    from openai import APIConnectionError, APITimeoutError
    _RETRYABLE_MODEL_EXC: tuple[type[BaseException], ...] = (APIConnectionError, APITimeoutError)
except ImportError:
    _RETRYABLE_MODEL_EXC = ()


def _is_retryable_model_error(e: BaseException) -> bool:
    """Decide whether a failed model call is worth retrying once.

    Retry on timeout / connection errors (both safe — chat completions have no
    side effects). Skip retry on programming errors like bad model id, malformed
    requests, etc., where the second attempt would just waste user time.
    """
    if isinstance(e, _RETRYABLE_MODEL_EXC):
        return True
    msg = str(e).lower()
    return "timeout" in msg or "timed out" in msg or "connection" in msg


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

Vision:
- The user may paste image file paths in their message; those images are auto-attached and visible to you when the loaded model supports vision. Analyze them when they're relevant to the request.

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
    # Auto-compaction kicks in after a turn whenever total_tokens / context_length
    # crosses this fraction. 0 disables auto-compaction (manual /compact still works).
    auto_compact_threshold: float = 0.8


COMPACTION_SUMMARY_PROMPT = """The conversation below is being compacted to free up context window.

Produce a concise factual summary that the assistant can use to continue helping the user. Cover:
- What the user has been trying to accomplish overall.
- Files created, modified, or deleted (paths only, with a one-line purpose each).
- Background processes started (PID + purpose if mentioned).
- Decisions made, constraints established, or important findings.
- Anything still in progress or unfinished.

Be brief and factual. Plain prose, no markdown headers. Skip pleasantries and tool-call mechanics — focus on substance the assistant needs to remember."""


@dataclass
class Agent:
    client: LMStudioClient
    config: AgentConfig
    messages: list[dict[str, Any]] = field(default_factory=list)
    on_event: Callable[[str, dict[str, Any]], None] | None = None
    plan_mode: bool = False
    # Tokens reported by the most recent response's usage.total_tokens, and the
    # loaded context window from LM Studio. Both None until first known.
    last_total_tokens: int | None = None
    context_length: int | None = None
    # Which warning thresholds we've already announced this conversation, so we
    # don't spam the user every turn after crossing them.
    _warned_levels: set[int] = field(default_factory=set)

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
        self._refresh_context_length()

    def _refresh_context_length(self) -> None:
        """Look up (or refresh from cache) the loaded context length for the
        active model. Best-effort — leaves context_length None on failure."""
        try:
            self.context_length = self.client.get_model_context_length(self.config.model)
        except Exception:
            self.context_length = None

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.on_event:
            self.on_event(kind, data)

    def reset(self) -> None:
        system = self.messages[0] if self.messages and self.messages[0]["role"] == "system" else None
        self.messages = [system] if system else []
        self.plan_mode = False
        self.state.todos = []
        # Fresh conversation — drop usage tracking and warning state.
        self.last_total_tokens = None
        self._warned_levels = set()

    def set_model(self, model: str) -> None:
        self.config.model = model
        # New model probably has a different context window — re-look it up.
        self._refresh_context_length()
        self._warned_levels = set()

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
        # Snapshot conversation length so we can roll back cleanly on Ctrl+C.
        # Without this, an interrupt mid-tool-call leaves the history with an
        # assistant turn containing tool_calls but no matching tool responses,
        # which is invalid against the chat-completions schema.
        snapshot_len = len(self.messages)
        self.state.touched_files.clear()
        attached = find_images(user_input, self.state.cwd)
        if attached:
            self._emit("images_attached", {
                "paths": [str(p) for _, p in attached],
            })
        self.messages.append(build_user_message(user_input, attached))

        try:
            final_text = self._run_tool_loop()
            # If the model call errored or hit the iteration cap, skip verification —
            # the user will retry, and we'll verify fresh on the next turn.
            if final_text.startswith(("[error]", "[stopped:")):
                return final_text
            result = self._verify_and_fix(final_text)
            # If usage crossed the auto-compact threshold, run compaction
            # before returning so the user sees what happened in-line and the
            # next turn starts with the freed context.
            self._maybe_auto_compact()
            return result
        except KeyboardInterrupt:
            # Hard reset: drop the user message, any partial assistant turn, and
            # any partial tool responses added during this turn.
            del self.messages[snapshot_len:]
            self._emit("interrupted", {})
            return "[interrupted]"

    def _run_tool_loop(self) -> str:
        """Drive the model<->tools loop until the model returns no tool calls or
        the iteration cap fires. Returns the final assistant text (which may be a
        sentinel like '[error] ...' or '[stopped: ...]' on failure)."""
        empty_retries = 0
        for iteration in range(1, self.config.max_tool_iters + 1):
            content, tool_calls, err = self._one_round_with_retry(iteration)
            if err is not None:
                self._emit("error", {"message": f"Model call failed: {err}"})
                # Pop the trailing user message so the user can retry. Works for
                # both the original user input and an injected verification prompt.
                if self.messages and self.messages[-1].get("role") == "user":
                    self.messages.pop()
                return f"[error] {err}"

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
                # Reasoning models (Qwen3, DeepSeek-R1, etc.) sometimes return
                # only internal thinking tokens that LM Studio strips before
                # sending, leaving content empty. Nudge up to MAX_EMPTY_RETRIES
                # times before giving up so the model still produces a reply.
                visible = _THINK_RE.sub("", content).strip()
                if not visible and empty_retries < MAX_EMPTY_RETRIES:
                    empty_retries += 1
                    self._emit("continuation_nudge", {
                        "attempt": empty_retries,
                        "max": MAX_EMPTY_RETRIES,
                    })
                    # Replace the empty assistant turn with a user nudge so the
                    # next round has a valid conversation structure to respond to.
                    self.messages.pop()
                    self.messages.append({
                        "role": "user",
                        "content": "Continue and provide your complete response to my request.",
                    })
                    continue
                return content or ""

            empty_retries = 0
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

        return self._wrap_up_after_cap()

    def _one_round_with_retry(self, iteration: int) -> tuple[str, list[dict[str, Any]], Exception | None]:
        """Run a single model round, retrying once on transient errors (timeout /
        connection failure). Returns (content, tool_calls, error)."""
        try:
            content, tool_calls = self._one_round(iteration)
            return content, tool_calls, None
        except Exception as e:
            if not _is_retryable_model_error(e):
                return "", [], e
            self._emit("model_retry", {
                "error": str(e),
                "delay": MODEL_RETRY_DELAY_SECS,
                "iteration": iteration,
            })
            time.sleep(MODEL_RETRY_DELAY_SECS)
            try:
                content, tool_calls = self._one_round(iteration)
                return content, tool_calls, None
            except Exception as e2:
                return "", [], e2

    def _wrap_up_after_cap(self) -> str:
        """Iteration cap fired with the model still wanting to call tools. Instead
        of returning a bare sentinel, inject a final system nudge and ask the
        model once more (with no tools available) for a human-readable summary of
        where it got to. Gives the user a graceful landing instead of an abrupt
        cut."""
        self._emit("iteration_cap", {"max_iters": self.config.max_tool_iters})
        self.messages.append({
            "role": "system",
            "content": (
                f"You have used your full budget of {self.config.max_tool_iters} tool iterations "
                "for this turn and cannot call any more tools. In one or two short sentences, "
                "summarize what you accomplished and what is still unfinished so the user can "
                "decide how to proceed. Do not attempt any further tool calls."
            ),
        })
        try:
            resp = self.client.chat(
                model=self.config.model,
                messages=self.messages,
                tools=None,
                temperature=self.config.temperature,
                stream=False,
            )
        except Exception as e:
            self._emit("error", {"message": f"Could not produce wrap-up summary: {e}"})
            return f"[stopped: max tool iterations reached; summary failed: {e}]"

        if not resp.choices:
            return "[stopped: max tool iterations reached]"
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            return "[stopped: max tool iterations reached]"
        self.messages.append({"role": "assistant", "content": content})
        self._emit("assistant_text", {"text": content})
        return content

    def _verify_and_fix(self, final_text: str) -> str:
        """Run cheap syntax checks on files the model touched this turn. If
        anything fails to parse, feed the errors back as a synthetic user turn
        so the model can self-correct — capped at MAX_VERIFICATION_CYCLES. Any
        errors that survive the auto-fix attempts are surfaced to the user."""
        # Nothing to check in plan mode (writes are blocked) or if no files
        # were touched.
        if self.plan_mode or not self.state.touched_files:
            return final_text

        touched_str = sorted(str(p) for p in self.state.touched_files)
        self._emit("verify_start", {"files": touched_str})

        for cycle in range(1, MAX_VERIFICATION_CYCLES + 1):
            errors = verify_touched_files(self.state)
            if not errors:
                self._emit("verify_pass", {"files": touched_str})
                return final_text

            error_payload = [{"path": str(p), "message": m} for p, m in errors]
            self._emit("verify_errors", {
                "cycle": cycle,
                "max_cycles": MAX_VERIFICATION_CYCLES,
                "errors": error_payload,
            })

            formatted = "\n".join(f"- {p}: {m}" for p, m in errors)
            self.messages.append({
                "role": "user",
                "content": (
                    "[automated verification] These files you just edited failed to parse:\n\n"
                    f"{formatted}\n\n"
                    "Fix them now with edit_file / write_file / multi_edit. "
                    "When finished, reply briefly with what you changed."
                ),
            })
            fix_text = self._run_tool_loop()
            if fix_text.startswith(("[error]", "[stopped:")):
                # Model failed mid-fix; surface what we have and bail out of the
                # verification loop. The original final_text is more useful to
                # the user than the error sentinel.
                break
            final_text = fix_text

        remaining = verify_touched_files(self.state)
        if remaining:
            self._emit("verify_failed", {
                "errors": [{"path": str(p), "message": m} for p, m in remaining],
            })
            final_text = (
                final_text
                + "\n\n[verification could not auto-fix the issues above — please review]"
            )
        else:
            self._emit("verify_pass", {"files": touched_str})
        return final_text

    # ---------- compaction ----------

    def compact(self, reason: str = "manual") -> dict[str, Any] | None:
        """Replace older turns with a model-generated summary. Keeps the system
        prompt and the most recent complete turns intact. Returns a result dict
        with stats for the UI, or None if there isn't enough history to make
        compaction worthwhile (or the summary call fails)."""
        split = self._find_compaction_split(keep_recent_turns=2)
        if split is None:
            self._emit("compact_skipped", {"reason": "not enough complete turns to compact"})
            return None

        to_compact = self.messages[1:split]
        kept_tail = self.messages[split:]
        old_count = len(self.messages)

        self._emit("compact_start", {
            "reason": reason,
            "compacting_messages": len(to_compact),
            "keeping_messages": len(kept_tail) + 1,  # +1 for the system prompt
        })

        # Ask the model to summarize. tools=None / stream=False to keep it
        # simple — no tool calls to dispatch on the summary turn itself.
        formatted = self._format_messages_for_summary(to_compact)
        summary_messages: list[dict[str, Any]] = [
            {"role": "system", "content": COMPACTION_SUMMARY_PROMPT},
            {"role": "user", "content": f"Conversation to summarize:\n\n{formatted}"},
        ]
        try:
            resp = self.client.chat(
                model=self.config.model,
                messages=summary_messages,
                tools=None,
                temperature=self.config.temperature,
                stream=False,
            )
        except Exception as e:
            self._emit("compact_failed", {"error": str(e)})
            return None

        if not resp.choices:
            self._emit("compact_failed", {"error": "no choices in summary response"})
            return None
        summary = (resp.choices[0].message.content or "").strip()
        if not summary:
            self._emit("compact_failed", {"error": "empty summary"})
            return None

        # Rewrite history: system + summary + kept tail.
        system = self.messages[0]
        summary_msg = {
            "role": "system",
            "content": (
                f"[compaction] Summary of {len(to_compact)} earlier message(s), "
                "replacing them to free context window:\n\n" + summary
            ),
        }
        self.messages = [system, summary_msg] + kept_tail

        # Conversation shape changed — token estimates are stale. The next
        # response's usage report will re-establish the count.
        self.last_total_tokens = None
        self._warned_levels = set()

        result = {
            "reason": reason,
            "messages_before": old_count,
            "messages_after": len(self.messages),
            "summary_chars": len(summary),
        }
        self._emit("compact_done", result)
        return result

    def _find_compaction_split(self, keep_recent_turns: int = 2) -> int | None:
        """Return the index AT which kept messages start: messages[1:split] gets
        summarized; messages[0] (system) and messages[split:] stay verbatim.

        The split lands immediately after a 'final assistant' (one with no
        tool_calls), so we never break a tool_call → tool_response chain.
        Returns None if there aren't enough complete turns to compact more than
        a few messages — not worth the round-trip below that."""
        final_assistants: list[int] = []
        for i, m in enumerate(self.messages):
            if m.get("role") == "assistant" and not m.get("tool_calls"):
                final_assistants.append(i)
        if len(final_assistants) <= keep_recent_turns:
            return None
        cutoff_idx = final_assistants[-keep_recent_turns - 1]
        split = cutoff_idx + 1
        # If we'd be summarizing fewer than 4 messages there's nothing to gain.
        if split - 1 < 4:
            return None
        return split

    def _format_messages_for_summary(self, messages: list[dict[str, Any]]) -> str:
        """Render an internal message list as compact text for the summary call.
        Tool outputs are truncated since they're typically the verbose part and
        we mostly care about WHAT was done, not the raw stdout."""
        lines: list[str] = []
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content", "") or ""
            # Multimodal user turns store content as a list of parts; flatten
            # to text so the summarizer doesn't see raw base64.
            if isinstance(content, list):
                flat: list[str] = []
                for p in content:
                    if not isinstance(p, dict):
                        continue
                    if p.get("type") == "text":
                        flat.append(p.get("text", ""))
                    elif p.get("type") == "image_url":
                        flat.append("[image]")
                content = " ".join(flat)
            if role == "tool":
                content = content[:500] + (" …[truncated]" if len(content) > 500 else "")
                tool_id = (m.get("tool_call_id") or "")[:8]
                lines.append(f"[tool result {tool_id}] {content}")
            elif role == "assistant":
                tc = m.get("tool_calls") or []
                if tc:
                    names = ", ".join(c.get("function", {}).get("name", "?") for c in tc)
                    lines.append(f"[assistant → tools: {names}] {content}")
                else:
                    lines.append(f"[assistant] {content}")
            elif role == "user":
                lines.append(f"[user] {content}")
            elif role == "system":
                lines.append(f"[system] {content}")
        return "\n\n".join(lines)

    def _maybe_auto_compact(self) -> None:
        """Run compaction automatically if the latest usage report puts us at
        or above the configured threshold. Disabled when threshold is 0 or we
        lack the data to know (no context length or no usage reported)."""
        if self.config.auto_compact_threshold <= 0:
            return
        if self.last_total_tokens is None or not self.context_length:
            return
        ratio = self.last_total_tokens / self.context_length
        if ratio < self.config.auto_compact_threshold:
            return
        self.compact(reason="auto")

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
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self._record_usage(getattr(usage, "total_tokens", None))
        msg = resp.choices[0].message
        content = msg.content or ""
        tcs = []
        for tc in getattr(msg, "tool_calls", None) or []:
            tcs.append({"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments or "{}"})
        self._emit("turn_end", {"iteration": iteration})
        if content:
            self._emit("assistant_text", {"text": content})
        return content, tcs

    def _record_usage(self, total_tokens: int | None) -> None:
        """Capture the token-usage report from a model response. Updates the
        agent's running count and emits a context_usage event so the UI can
        update the status line and fire threshold warnings."""
        if total_tokens is None or not isinstance(total_tokens, int):
            return
        self.last_total_tokens = total_tokens
        ratio = (total_tokens / self.context_length) if self.context_length else None
        self._emit("context_usage", {
            "tokens": total_tokens,
            "max": self.context_length,
            "ratio": ratio,
        })
        # Fire one-time warnings at 80% / 90% / 95% so the user is never
        # surprised by silent truncation. We only fire each level once per
        # conversation (cleared on /clear, /compact, /model switch).
        if ratio is None:
            return
        for level in (80, 90, 95):
            if ratio * 100 >= level and level not in self._warned_levels:
                self._warned_levels.add(level)
                self._emit("context_warning", {"level": level, "ratio": ratio})

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
        try:
            for event in stream:
                # The final chunk (when stream_options.include_usage is set)
                # carries usage with empty choices. Capture and skip rendering.
                usage = getattr(event, "usage", None)
                if usage is not None:
                    self._record_usage(getattr(usage, "total_tokens", None))
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
        finally:
            # Always release the HTTP connection and tell the UI to take down
            # the live streaming panel — both on normal completion AND on
            # Ctrl+C, where the exception still propagates out after cleanup.
            try:
                stream.close()
            except Exception:
                pass
            self._emit("turn_end", {
                "iteration": iteration,
                "elapsed": time.monotonic() - start,
                "tokens": token_count,
            })

        content = "".join(content_parts)
        if content:
            self._emit("assistant_text", {"text": content})
        tcs = [tc_acc[i] for i in sorted(tc_acc.keys())]
        return content, tcs
