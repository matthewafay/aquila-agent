# Aquila Agent — Local AI Coding Agent for LM Studio

A terminal AI agent harness that talks to a local **LM Studio** server. Acts like a stripped-down Claude Code — same kind of file / shell / web tool loop, but powered by whatever model you have loaded locally. The CLI ships as `aquila`.

### At a glance

- **14 callable tools** across file editing, shell (foreground + background), search/grep, web, and task tracking
- **Plan mode** — read-only investigation, then `/approve` to execute
- **Streaming UI** — live token counter, elapsed timer, Markdown rendering of model output, syntax-highlighted diffs for edits, JSON panels for tool calls
- **Background processes** — the model can launch dev servers / watchers; they get logged to `.aquila/bg_*.log` and survive across agent turns, with `/processes` and `/stop <pid>` to manage them
- **Structured todos** — the model maintains a checklist for multi-step work, rendered as a live panel
- **Hot model switching** — `/model <substring>` swaps to any other model currently loaded in LM Studio mid-conversation
- **Cross-platform shell** — PowerShell on Windows, bash on Unix; same tool name either way
- **Persistent REPL history** — your previous prompts are in `~/.aquila_history`, with slash-command tab completion
- **OpenAI-compatible** — uses the official `openai` SDK pointed at LM Studio, so the agent loop also works against any other OpenAI-compatible local server (Ollama's OpenAI shim, llama.cpp server, vLLM, etc.) with `--base-url`

## Setup

**Requirements:** Python 3.10+ and a UTF-8 terminal.

Aquila is a client — the model lives in [LM Studio](https://lmstudio.ai/), which has to be running with a tool-capable model loaded *before* `aquila` starts.

### 1. Get LM Studio serving a model

Install LM Studio, then in the **Discover** tab download a model tagged **"Tool Use"** (Qwen 2.5 Coder Instruct, Llama 3.1 Instruct, and Hermes 3 are solid defaults). Load it via the model dropdown, then open **Developer → Start Server** (default `http://localhost:1234`). Quick sanity check:

```powershell
curl http://localhost:1234/v1/models
```

If that returns JSON listing your model, you're good. If not, fix LM Studio first — `aquila` calls the same endpoint on launch and will exit with code 2 if it's unreachable.

### 2. Install aquila

```powershell
git clone https://github.com/matthewafay/aquila-agent.git
cd aquila-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1       # macOS/Linux: source .venv/bin/activate
pip install -e .
```

### 3. Run

```powershell
aquila
```

You'll see a banner with the model and working directory, then a `»` prompt. The directory you launch from becomes the agent's working directory — `cd` into the project you want it to operate on, or pass `--cwd <path>`.

## CLI flags

```powershell
# Interactive REPL — uses the first loaded model
aquila

# Pick a specific model (substring match works, e.g. "qwen" → "qwen2.5-coder-7b-instruct")
aquila --model qwen

# One-shot prompt, then exit
aquila "list the python files under src and summarize each"

# Point at a remote LM Studio (another machine on the network)
aquila --base-url http://192.168.1.50:1234/v1

# Different working directory for file/shell tools
aquila --cwd C:\Code\some-project

# Tuning
aquila --temperature 0.4 --max-iters 40

# Raise the model-call timeout if you hit "Model call failed: timed out" on
# long generations (default 1200s; transient timeout/connection errors are
# already auto-retried once before the turn bails).
aquila --request-timeout 1800
```

Environment overrides (so you don't have to pass flags every time):

- `AQUILA_BASE_URL` — default base URL (e.g. `http://localhost:1234/v1`)
- `AQUILA_API_KEY` — default API key (LM Studio accepts anything; defaults to `lm-studio`)
- `AQUILA_REQUEST_TIMEOUT` — default per-request timeout in seconds for model calls (default `1200`)

## REPL commands

Anything not starting with `/` is sent to the model.

| command              | what it does                                                |
| -------------------- | ----------------------------------------------------------- |
| `/help`              | show command list                                           |
| `/models`            | list models currently loaded in LM Studio                   |
| `/model <id>`        | switch the active model (substring match works)             |
| `/clear`             | wipe chat history (keeps the system prompt)                 |
| `/cwd [path]`        | show or change the working directory for tools              |
| `/history`           | dump the raw message log                                    |
| `/plan [task]`       | enter **plan mode** — read-only tools only                  |
| `/approve` / `/exec` | leave plan mode and tell the model to execute the plan      |
| `/cancel`            | leave plan mode without executing                           |
| `/todos`             | show the current task list                                  |
| `/processes`         | list background processes started by the agent              |
| `/stop <pid>`        | terminate a background process by PID                       |
| `/queue [prompt]`    | queue a follow-up prompt; no-arg shows the queue, `clear` empties it |
| `/exit`, `/quit`     | leave (Ctrl+D also works)                                   |

### Interrupting a turn

Press **Ctrl+C** while the model is working to cancel the current turn. The agent rolls back the conversation history to before your prompt (so there are no dangling `tool_calls` without responses), then shows a `redirect » ` prompt:

- **Empty Enter** — just cancel, back to the normal `»` prompt.
- **Type a new prompt** — submits it as the next turn (steering away from whatever the model was doing).

Files written to disk and background processes spawned before the interrupt are NOT rolled back — only the model's memory of them is. The model will start the next turn fresh; if it needs to see what's on disk, it'll `read_file` like normal.

### Queueing follow-up prompts

Type `/queue <prompt>` one or more times *before* sending your main prompt. The main prompt runs first; when it finishes normally, queued prompts fire in order. `/queue` with no argument lists what's queued; `/queue clear` empties it. The queue is cleared automatically if you Ctrl+C mid-turn — queued prompts were authored against context that just got rolled back, so re-running them blind would be wrong.

True mid-turn queueing (typing while the model is streaming) is a known TODO — it needs an asyncio refactor of the REPL so prompt_toolkit's input and Rich's Live display can coexist.

### Plan mode

`/plan` puts the agent in a read-only sandbox: it can `read_file`, `list_dir`, `search_files`, `grep`, `web_search`, `web_fetch`, `list_processes`, and `todo_write`, but **not** write files or run shell commands. Use it to ask "what would you change to do X?" and get a step-by-step proposal before any code moves. `/approve` (or `/exec`) drops back to full tool access and tells the model to carry out the plan. `/cancel` exits plan mode without doing anything.

## Tools the model can call

| tool                   | purpose                                                                |
| ---------------------- | ---------------------------------------------------------------------- |
| `read_file`            | UTF-8 file read (up to 400 KB)                                         |
| `write_file`           | create / overwrite a file (auto-creates parent dirs)                   |
| `edit_file`            | replace one unique substring in a file                                 |
| `multi_edit`           | apply many (old → new) replacements to one file atomically             |
| `list_dir`             | non-recursive directory listing                                        |
| `search_files`         | recursive glob, e.g. `**/*.py`                                         |
| `grep`                 | recursive regex search across text files                               |
| `run_shell`            | foreground PowerShell / bash command (120 s timeout)                   |
| `run_shell_background` | spawn a long-running process (dev servers, watchers); logs to `.aquila/`|
| `list_processes`       | list background processes the agent started                            |
| `stop_process`         | terminate one of those background processes by PID                     |
| `web_search`           | DuckDuckGo search                                                      |
| `web_fetch`            | GET a URL, return readable text (HTML stripped)                        |
| `todo_write`           | maintain a structured task list for the current request                |

All relative paths resolve against the current `/cwd`.

### Tool details & limits

**File tools**

- `read_file(path)` — UTF-8 only. Files larger than **400 KB** are rejected with an error rather than truncated, to keep the conversation context predictable. Binary files (anything that isn't valid UTF-8) return an explicit error.
- `write_file(path, content)` — creates parent directories automatically. Always overwrites. The UI renders the proposed content as a syntax-highlighted panel before the write happens (language auto-detected from the file extension — Python, TS/JS/JSX, HTML, CSS, JSON, YAML, TOML, Markdown, Bash, PowerShell, SQL, Rust, Go, Java, C/C++, Ruby, PHP, Swift, Kotlin, XML, Lua).
- `edit_file(path, old, new)` — replaces **exactly one** occurrence of `old`. If `old` isn't unique in the file (or doesn't appear at all) the edit fails and the file is untouched. The UI shows a unified diff for the change.
- `multi_edit(path, edits=[{old, new}, ...])` — batch of edits applied in order. Each `old` must be unique in the file at the moment its edit runs. **All-or-nothing**: if any edit fails, the file is not modified. Up to 8 edits are rendered as inline diffs in the UI (any beyond that are collapsed to a "+N more" line).

**Search tools**

- `list_dir(path=".")` — non-recursive. Prints `DIR ` / `FILE` markers plus byte sizes.
- `search_files(pattern, path=".")` — recursive glob (`**/*.py`-style). Capped at **500 matches**.
- `grep(pattern, path=".")` — recursive regex search over text files. Uses **ripgrep** automatically if `rg` is on `PATH` (respects `.gitignore`, skips hidden files, much faster on large repos); otherwise falls back to a pure-Python walk that searches everything and is capped at **5000 files**. Either way the result is capped at **300 matching lines** and emitted as `path:line: content`. If ripgrep rejects a regex feature it doesn't support (e.g. lookaround), the Python fallback runs instead.

**Shell tools**

- `run_shell(command)` — foreground. 120 s timeout. Returns `STDOUT` / `STDERR` / `EXIT` sections. Output truncated to ~16 KB. PowerShell flags used: `-NoProfile -NonInteractive -Command`; bash uses `bash -lc`.
- `run_shell_background(command)` — spawns detached, returns immediately with the PID. Logs combined stdout/stderr to `<cwd>/.aquila/bg_<hash>.log`. On Windows it uses `CREATE_NEW_PROCESS_GROUP` so the child survives independently of the agent's console.
- `list_processes()` — shows every background PID with `running` or `exited(<code>)` status. Dead PIDs are reaped after listing.
- `stop_process(pid)` — sends `terminate()`, then `kill()` after a 5 s grace period.

**Web tools**

- `web_search(query, max_results=5)` — DuckDuckGo via the `duckduckgo-search` library. Returns title / URL / snippet for each hit.
- `web_fetch(url)` — `httpx` GET with 30 s timeout, follows redirects, sends `User-Agent: aquila/0.1`. HTML responses go through BeautifulSoup, get stripped of `<script>` / `<style>` / `<noscript>`, and are flattened to text. Non-HTML responses are returned as-is. Output truncated to ~16 KB.

**Task tracking**

- `todo_write(todos=[{content, activeForm, status}, ...])` — replaces the entire todo list each call. `status` must be one of `pending`, `in_progress`, `completed`. At most one item may be `in_progress` at a time (enforced server-side). `content` is the imperative ("Create index.html"), `activeForm` is the present continuous ("Creating index.html"). The UI renders the list as a bordered panel with `✓` / `▶` / `○` markers and a "done / total" counter; the system prompt instructs the model to call this at the start of any 3+ step task and to update it as work progresses.

## How long-running servers work

When the model needs to start something that doesn't exit on its own (`npm run dev`, `vite`, `python -m http.server`, etc.) it uses `run_shell_background` instead of `run_shell`. That:

1. spawns the process detached from the agent,
2. writes its combined stdout/stderr to `<cwd>/.aquila/bg_<hash>.log`,
3. returns the PID immediately so the agent can keep working.

You can inspect them yourself with `/processes` and kill them with `/stop <pid>`. They are also cleaned up when you exit the REPL.

## How the agent loop works

Each user turn runs a bounded tool-call loop (default **25 iterations**, configurable via `--max-iters`):

1. The agent sends the full message history plus the active tool schemas to LM Studio.
2. The response is streamed back. Text deltas update a live Markdown panel; tool-call fragments are accumulated by index.
3. If the model emitted any tool calls, each is dispatched locally:
   - Arguments are JSON-decoded; bad JSON returns a structured error to the model so it can self-correct.
   - In plan mode, write/shell tools are blocked at the dispatcher level — the model gets `ERROR: '<tool>' is blocked in plan mode` and has to propose instead.
   - Tool output is truncated to ~16 KB before being appended to the conversation as a `role: "tool"` message.
4. The loop continues until the model produces a turn with **no tool calls** (final answer) or hits the iteration cap.
5. **Post-turn verification.** Once the model declares it's done, the agent runs cheap syntax checks on every file it just touched (`.py` via `compile()`, `.json` via `json.loads`, `.toml` via `tomllib`). If anything fails to parse, the errors are fed back to the model as a synthetic user turn and it gets a chance to self-correct — capped at **2 auto-fix attempts**. Anything still broken after that is surfaced as a red `needs your attention` panel so you can review. Other file types are skipped (no false positives on languages we don't have a stdlib parser for), and the phase is a no-op in plan mode.

**Resilience.** Two common failure modes are handled inline instead of bailing the turn:

- **Transient model-call errors** (timeout / connection failure from LM Studio) are **auto-retried once** with a 2-second pause. You'll see a yellow `⟳ model call failed … retrying once` line; if the retry also fails, the turn ends with the error and your prompt is preserved for re-submission. The hard timeout is `--request-timeout` / `AQUILA_REQUEST_TIMEOUT` (default 1200s). Non-transient errors (bad model id, malformed request) skip the retry.
- **Iteration-cap exhaustion.** Instead of returning a bare `[stopped: max tool iterations reached]`, the agent sends the model one final call *without* tools asking for a one-sentence summary of where it got to. You get a graceful landing — "I created X and Y but still need to do Z" — and can decide whether to follow up or bump `--max-iters`.

The system prompt explicitly tells the model to **do the work itself** rather than instruct the user (e.g. it will run `npm install` via `run_shell` instead of telling you to), to use `run_shell_background` for anything long-running, and to keep the todo list updated for multi-step jobs.

## What you see in the terminal

The Rich-based renderer surfaces the agent's internal events as distinct visual elements:

- **Live streaming panel** — model output rendered as Markdown in real time, with a footer showing iteration number, elapsed seconds, and token count.
- **Tool-call panels** (cyan border) — JSON arguments for each tool call, pretty-printed. Special cases: `write_file` shows syntax-highlighted file content; `edit_file` and `multi_edit` show unified diffs (magenta border).
- **Tool-result panels** (green border) — truncated to ~1500 chars in the display (full result still goes to the model). Writes/edits collapse to a one-line `→ Wrote N chars` confirmation.
- **Todos panel** (blue border) — refreshed every time the model calls `todo_write`.
- **Error lines** (red) — surfaced on model-call failures or invalid commands; the user message is popped so you can retry without re-typing.

## Notes & gotchas

- **No confirmation on shell commands.** `run_shell` and `run_shell_background` execute whatever the model decides to run. Don't aim this at a directory you can't afford to lose, and consider running aggressive models inside a VM or a throwaway folder. Use `/plan` first if you're not sure what a model will do.
- **Tool-calling support varies.** If a model chats back without ever invoking tools, swap to one that supports OpenAI-style `tools` (Qwen 2.5 Instruct, Llama 3.1 8B Instruct, Hermes 3, etc.).
- **Context windows.** Conversation history grows until `/clear`. On small-context models, long sessions will start dropping the earliest turns — clear when in doubt. The tool-call loop is also capped at 25 iterations per user turn by default (`--max-iters` to raise it).
- **Windows consoles.** The CLI forces UTF-8 on stdout/stderr so Rich's glyphs survive on legacy `cmd.exe`. If you still see mojibake, run inside Windows Terminal or PowerShell 7+.
- **`.aquila/` directory.** Background-process logs land in `<cwd>/.aquila/`. Add it to `.gitignore` in any project you point aquila at.
- **REPL history.** Your typed prompts persist to `~/.aquila_history` across sessions (prompt_toolkit `FileHistory`). Delete the file to wipe it.
- **Network access.** The web tools hit DuckDuckGo and arbitrary URLs directly — no proxy support, no robots.txt enforcement, no auth.
- **No sandboxing of `cwd`.** Tools resolve relative paths against `state.cwd`, but absolute paths are honored. A model that asks to read `C:\Windows\...` will get to read it. Run inside a VM if that's a concern.

## Example sessions

**Build a static site from scratch**

```text
» build me a single-page resume site for "Jane Doe, ML engineer" with a dark theme,
  and serve it locally so I can preview it
```
The model writes `todo_write` with a plan, creates `index.html` / `style.css` (you see the syntax-highlighted file contents in cyan panels), launches `python -m http.server 8000` via `run_shell_background`, and reports the PID + the local URL. You can `/stop <pid>` when done.

**Investigate before changing anything**

```text
» /plan refactor the auth middleware so session tokens aren't logged
```
In plan mode the model can only `read_file`, `grep`, `list_dir`, etc. It produces a numbered plan describing each file it would touch. `/approve` exits plan mode and the model carries it out; `/cancel` discards the plan.

**One-shot from the shell**

```powershell
aquila "summarize every TODO comment in this repo"
```
The agent runs `grep`, formats the findings, and exits.

**Swap models mid-conversation**

```text
» /models
» /model llama
```
The conversation history is preserved; the next turn just goes to the new model. Useful for sending speculative drafting to a fast model and final edits to a stronger one without losing context.

## Why the name

**Aquila** was the eagle standard carried at the head of every Roman legion. It was the rallying point in battle and the legal embodiment of the legion itself — losing the aquila was a disaster a legion might never recover from. This harness plays the same role for a local model: it's the standard the agent rallies around, the thing that turns a loose LM Studio process into a coordinated unit that can drive the terminal, edit files, and run servers on your behalf.

## Project layout

```
aquila/
├── __init__.py
├── __main__.py     # CLI entry: arg parsing, model selection, REPL launch
├── agent.py        # tool-call loop, system prompt, plan mode, streaming
├── client.py       # thin OpenAI-SDK wrapper pointed at LM Studio
├── state.py        # agent state: cwd, todos, background processes
├── tools.py        # tool implementations + JSON schemas
└── ui.py           # Rich event renderer + prompt_toolkit REPL
pyproject.toml
README.md
```

## License

MIT — do whatever, no warranty.
