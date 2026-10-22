"""Readable subtitle units with exact text coverage and genuine alignment times."""
from __future__ import annotations

import unicodedata
from typing import Any

from asr_server.errors import AsrError


def _normalized(value: str) -> str:
    return "".join(char for char in value.casefold() if not char.isspace()
                   and not unicodedata.category(char).startswith("P"))


def build_reading_segments(text: str, timestamps: list[dict[str, Any]], *,
                           prefix: str = "segment", max_seconds: float = 6,
                           max_chars: int = 28) -> list[dict[str, Any]]:
    """Organize aligned text; never interpolate or invent word timings.

    Missing/contradictory alignment is a controlled error, not an invitation to
    drop words. All punctuation and inter-word spaces remain attached to text.
    """
    if not text.strip():
        return []
    if not timestamps:
        raise AsrError(422, "alignment_incomplete", "readable subtitles require genuine aligned units")
    normalized = ""
    positions = []
    for index, char in enumerate(text):
        part = _normalized(char)
        normalized += part
        positions.extend([index] * len(part))
    spans: list[tuple[int, int, dict[str, Any]]] = []
    cursor = 0
    previous_start = -1.0
    for item in timestamps:
        token = _normalized(str(item["text"]))
        if not token:
            continue
        start, end = float(item["start"]), float(item["end"])
        if start < 0 or end < start or start < previous_start:
            raise AsrError(422, "alignment_incomplete", "subtitle alignment is not ordered")
        previous_start = start
        if not normalized.startswith(token, cursor):
            raise AsrError(422, "alignment_text_mismatch", "aligned units do not cover this transcript version")
        spans.append((positions[cursor], positions[cursor + len(token) - 1] + 1, item))
        cursor += len(token)
    if cursor != len(normalized) or not spans:
        raise AsrError(422, "alignment_incomplete", "aligned units do not cover all transcript words")
    units: list[tuple[str, dict[str, Any]]] = []
    for number, (start_char, _end_char, item) in enumerate(spans):
        left = 0 if number == 0 else start_char
        right = spans[number + 1][0] if number + 1 < len(spans) else len(text)
        units.append((text[left:right], item))
    result: list[dict[str, Any]] = []
    group: list[tuple[str, dict[str, Any]]] = []

    def finish() -> None:
        if not group:
            return
        content = "".join(part for part, _item in group).strip()
        if content:
            result.append({"id": f"{prefix}-{len(result):05d}", "index": len(result),
                "start": float(group[0][1]["start"]), "end": float(group[-1][1]["end"]),
                "text": content, "timestamps": [item for _part, item in group],
                "timestamp_source": str(group[0][1].get("source", "unknown"))})
        group.clear()

    for part, item in units:
        if group:
            span = float(item["end"]) - float(group[0][1]["start"])
            gap = float(item["start"]) - float(group[-1][1]["end"])
            count = len(_normalized("".join(p for p, _i in group) + part))
            if span > max_seconds or gap >= .8 or count > max_chars:
                finish()
        group.append((part, item))
        stripped = part.rstrip()
        if stripped.endswith(("。", "！", "？", "!", "?", ";", "；")):
            finish()
        elif stripped.endswith(("，", ",", "、")) and len(_normalized("".join(p for p, _i in group))) >= 12:
            finish()
    finish()
    # Whitespace may be reformatted between cues. No lexical unit may vanish.
    if _normalized("".join(segment["text"] for segment in result)) != normalized:
        raise AsrError(422, "alignment_text_mismatch", "subtitle grouping lost transcript content")
    return result
