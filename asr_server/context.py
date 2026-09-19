from __future__ import annotations

import json
import re
from dataclasses import dataclass

from asr_server.errors import AsrError


MAX_CONTEXT_CHARS = 4_000
MAX_CONTEXT_BODY_CHARS = 3_000
MAX_HOTWORDS = 100
MAX_HOTWORD_CHARS = 120
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class PreparedContext:
    prompt: str
    requested: bool
    input_chars: int
    used_chars: int
    hotword_count: int
    truncated: bool
    warnings: tuple[str, ...]

    def request_summary(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "input_chars": self.input_chars,
            "used_chars": self.used_chars,
            "hotword_count": self.hotword_count,
            "truncated": self.truncated,
        }


def parse_hotwords(hotwords: str | None) -> list[str]:
    return _parse_hotwords(hotwords)[0]


def _parse_hotwords(hotwords: str | None) -> tuple[list[str], bool]:
    if hotwords is None or not hotwords.strip():
        return [], False
    stripped = hotwords.strip()
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise AsrError(400, "bad_request", "hotwords must be a JSON string array or comma-separated string") from exc
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise AsrError(400, "bad_request", "hotwords JSON value must be an array of strings")
        candidates = parsed
    else:
        candidates = stripped.split(",")

    normalized: list[str] = []
    seen: set[str] = set()
    item_truncated = False
    for candidate in candidates:
        item = _normalize_text(candidate).strip()
        if not item:
            continue
        if len(item) > MAX_HOTWORD_CHARS:
            item = item[:MAX_HOTWORD_CHARS].rstrip()
            item_truncated = True
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)
    return normalized, item_truncated


def prepare_context(context: str, hotwords: str | None) -> PreparedContext:
    normalized_context = _normalize_text(context).strip()
    normalized_hotwords, hotword_item_truncated = _parse_hotwords(hotwords)
    input_chars = len(normalized_context) + sum(len(item) for item in normalized_hotwords)
    requested = bool(normalized_context or normalized_hotwords)
    warnings: list[str] = []
    truncated = False

    if hotword_item_truncated:
        warnings.append("context_truncated:hotword_item")
        truncated = True

    if len(normalized_context) > MAX_CONTEXT_BODY_CHARS:
        normalized_context = normalized_context[:MAX_CONTEXT_BODY_CHARS].rstrip()
        warnings.append("context_truncated:body")
        truncated = True
    if len(normalized_hotwords) > MAX_HOTWORDS:
        normalized_hotwords = normalized_hotwords[:MAX_HOTWORDS]
        warnings.append("context_truncated:hotword_count")
        truncated = True

    parts = ["Reference vocabulary only. Transcribe only speech present in the audio; do not invent listed terms."]
    if normalized_context:
        parts.append("Context data:\n" + normalized_context)
    used_hotwords: list[str] = []
    prompt = "\n\n".join(parts) if requested else ""
    for item in normalized_hotwords:
        keyword_block = "Keywords:\n" + "\n".join(f"- {value}" for value in [*used_hotwords, item])
        candidate = "\n\n".join([*parts, keyword_block])
        if len(candidate) > MAX_CONTEXT_CHARS:
            warnings.append("context_truncated:total")
            truncated = True
            break
        used_hotwords.append(item)
        prompt = candidate

    return PreparedContext(
        prompt=prompt,
        requested=requested,
        input_chars=input_chars,
        used_chars=len(prompt),
        hotword_count=len(used_hotwords),
        truncated=truncated,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _normalize_text(value: str) -> str:
    return _CONTROL_CHARS.sub(" ", value.replace("\r\n", "\n").replace("\r", "\n"))
