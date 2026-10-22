from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from asr_server.audio.subtitles import build_reading_segments
from asr_server.errors import AsrError


def _raw_span(text: str, raw: str, start: int, end: int) -> str:
    """Map a display slice to its original audit text without changing either."""
    intervals: list[tuple[int, int]] = []
    for tag, a_start, a_end, b_start, b_end in SequenceMatcher(a=text, b=raw, autojunk=False).get_opcodes():
        if a_end <= start or a_start >= end:
            continue
        if tag == "equal":
            intervals.append((b_start + max(start - a_start, 0), b_end - max(a_end - end, 0)))
        else:
            intervals.append((b_start, b_end))
    return raw[min(item[0] for item in intervals):max(item[1] for item in intervals)] if intervals else ""


def reading_segments(chunks: list[dict[str, object]]) -> list[dict[str, object]]:
    """Use the shared strict grouping algorithm and attach raw-text audit slices."""
    output: list[dict[str, object]] = []
    for chunk in chunks:
        text, raw = str(chunk["text"]), str(chunk["raw_text"])
        values = chunk.get("timestamps", [])
        if not isinstance(values, list) or not values:
            output.append({**chunk, "index": len(output), "id": f"segment-{len(output):05d}", "chunk_index": chunk["index"]})
            continue
        items: list[dict[str, Any]] = values
        char_limit = 28 if any("\u4e00" <= char <= "\u9fff" for char in text) else 72
        try:
            groups = build_reading_segments(text, items, max_chars=char_limit)
        except AsrError:
            output.append({**chunk, "index": len(output), "id": f"segment-{len(output):05d}", "chunk_index": chunk["index"], "reading_segmentation": "unmapped_alignment"})
            continue
        cursor = 0
        for group in groups:
            content = str(group["text"])
            left = text.find(content, cursor)
            if left < 0:
                left = cursor
            right = left + len(content)
            output.append({**group, "index": len(output), "id": f"segment-{len(output):05d}",
                           "chunk_index": chunk["index"], "raw_text": _raw_span(text, raw, left, right).strip()})
            cursor = right
    return output
