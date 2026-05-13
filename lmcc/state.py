from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TodoItem:
    content: str
    activeForm: str
    status: str  # "pending" | "in_progress" | "completed"


@dataclass
class AgentState:
    """Mutable state shared between Agent and its tools."""

    cwd: Path
    bg_processes: dict[int, subprocess.Popen] = field(default_factory=dict)
    todos: list[TodoItem] = field(default_factory=list)

    def cleanup(self) -> None:
        for p in list(self.bg_processes.values()):
            try:
                p.terminate()
            except Exception:
                pass
