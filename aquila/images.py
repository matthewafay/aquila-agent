"""Image attachment support for multimodal models.

Scans user input for image-file-looking tokens, validates them against the
filesystem, and builds OpenAI-compatible multimodal messages (text + image_url
content parts) suitable for vision-language models served by LM Studio.

Text-only flow is preserved: if no images are detected, build_user_message
returns the original ``{"role": "user", "content": "<text>"}`` shape so that
non-vision models keep working unchanged.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any


IMG_EXTS: set[str] = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

_MIME: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

# Three alternatives: "double-quoted path", 'single-quoted path', or a bare
# unquoted token. The bare alternative is bounded by (?<!\S) on the left and
# (?=$|\s|[,.;:!?)]) on the right so we don't gobble surrounding punctuation
# like the leading "(" in "(bar.png)" or a trailing comma.
_IMG_RE = re.compile(
    r'"(?P<dq>[^"\r\n]+?\.(?:png|jpe?g|gif|webp|bmp))"'
    r"|'(?P<sq>[^'\r\n]+?\.(?:png|jpe?g|gif|webp|bmp))'"
    r"|(?<!\S)(?P<bare>\S+?\.(?:png|jpe?g|gif|webp|bmp))(?=$|\s|[,;:!?)])",
    re.IGNORECASE,
)


def find_images(text: str, cwd: Path) -> list[tuple[str, Path]]:
    """Scan ``text`` for image-path-looking tokens and return the ones that
    resolve to an existing file.

    Returns a list of (matched_substring, resolved_path) pairs. The matched
    substring is the exact slice from ``text`` (including any surrounding
    quotes) so callers can do literal replacements. Bare relative paths are
    resolved against ``cwd``.
    """
    found: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for m in _IMG_RE.finditer(text):
        raw = m.group("dq") or m.group("sq") or m.group("bare")
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = cwd / candidate
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError):
            continue
        if resolved in seen:
            continue
        if not resolved.is_file():
            continue
        if resolved.suffix.lower() not in IMG_EXTS:
            continue
        seen.add(resolved)
        found.append((m.group(0), resolved))
    return found


def encode_image(path: Path) -> str:
    """Read ``path`` and return an OpenAI-compatible data URL string."""
    mime = _MIME.get(path.suffix.lower(), "application/octet-stream")
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_user_message(
    text: str,
    attached: list[tuple[str, Path]],
) -> dict[str, Any]:
    """Build the user message dict to append to ``messages``.

    With no attachments, returns the plain ``{"role": "user", "content": str}``
    shape so text-only models see no behavior change. With attachments, returns
    the multimodal content-parts shape and rewrites each detected path token
    in the user's text to a short ``[image: name]`` marker so the model still
    has positional context for what each image refers to.
    """
    if not attached:
        return {"role": "user", "content": text}

    cleaned = text
    for token, path in attached:
        cleaned = cleaned.replace(token, f"[image: {path.name}]", 1)

    parts: list[dict[str, Any]] = []
    if cleaned.strip():
        parts.append({"type": "text", "text": cleaned})
    for _, path in attached:
        parts.append({
            "type": "image_url",
            "image_url": {"url": encode_image(path)},
        })
    return {"role": "user", "content": parts}
