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

## Prerequisites — set up LM Studio first

**Aquila does nothing on its own.** It is a client; the actual model lives in LM Studio. You must have LM Studio installed, running in **server mode**, and with a tool-calling model loaded **before** you start `aquila`. If any of those three things isn't true, `aquila` will exit with an error on launch.

1. **Install LM Studio.** Download from [lmstudio.ai](https://lmstudio.ai/) (Windows / macOS / Linux builds). Install and launch it.
2. **Download a tool-calling-capable model.** Open the **Discover** (search) tab inside LM Studio and pull one of:
   - Qwen 2.5 Instruct (7B / 14B / 32B) — recommended starting point
   - Qwen 2.5 Coder Instruct
   - Llama 3.1 / 3.2 Instruct
   - Mistral Small Instruct
   - Hermes 3
   - Anything else LM Studio tags as supporting **"Tool Use"** in the model card
   Models without tool-calling support will just chat back at you and never touch files.
3. **Load the model.** Go to the **Chat** or **Developer** tab and select the model from the top dropdown. Wait until it shows as fully loaded into memory (you'll see VRAM/RAM usage stabilize).
4. **Start the server.** Open the **Developer** tab on the left sidebar, then click **Start Server**. The default is `http://localhost:1234` — leave it there unless you have a conflict. You should see "Server running" with a green indicator.
5. **(Optional) Verify it's reachable.** From a terminal:
   ```powershell
   curl http://localhost:1234/v1/models
   ```
   You should get back JSON listing the loaded model(s). If this fails, `aquila` will fail too.

Only after all five steps should you run `aquila`. The CLI's first action is to call `/v1/models` against LM Studio; if the server is down or no model is loaded, it prints a hint and exits with code 2.

### Other requirements

- Python **3.10+**
- A terminal that handles UTF-8 (Windows Terminal, PowerShell 7+, any modern macOS/Linux terminal)

## Install

Clone and install in editable mode:

```powershell
git clone https://github.com/<you>/aquila-agent.git
cd aquila-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -e .
```

This installs the `aquila` console script and pulls in the dependencies declared in [pyproject.toml](pyproject.toml):

- `openai>=1.40` — OpenAI Python SDK, pointed at the LM Studio server
- `rich>=13.7` — terminal rendering (panels, syntax highlighting, live streaming)
- `prompt_toolkit>=3.0` — REPL input with history + slash-command completion
- `duckduckgo-search>=6.2` — the `web_search` tool
- `httpx>=0.27` — the `web_fetch` tool and a connectivity ping
- `beautifulsoup4>=4.12` — HTML → text for `web_fetch`

If you'd rather not install the script entry point:

```powershell
pip install openai rich prompt_toolkit duckduckgo-search httpx beautifulsoup4
python -m aquila
```

## Quick start

1. Open LM Studio → **Developer** tab → **Start Server** (port `1234`).
2. Load a tool-calling-capable model in LM Studio.
3. From the terminal:

```powershell
aquila
```

You should see a banner showing the model and the working directory, then a `»` prompt.

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
```

Environment overrides (so you don't have to pass flags every time):

- `AQUILA_BASE_URL` — default base URL (e.g. `http://localhost:1234/v1`)
- `AQUILA_API_KEY` — default API key (LM Studio accepts anything; defaults to `lm-studio`)

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
| `/exit`, `/quit`     | leave (Ctrl+D also works)                                   |

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
- `grep(pattern, path=".")` — recursive Python regex over text files. Skips binary / non-UTF-8 files silently. Capped at **300 matching lines** and walks at most **5000 files**. Output format: `path:line: content`.

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
