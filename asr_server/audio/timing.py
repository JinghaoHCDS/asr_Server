from __future__ import annotations

import re
from typing import Protocol, Sequence


class ChunkWindow(Protocol):
    @property
    def start(self) -> float: ...

    @property
    def end(self) -> float: ...


_PUNCTUATION = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)


def timing_analysis(
    timestamps: list[dict[str, object]],
    *,
    transcript: str,
    duration: float,
    chunks: Sequence[ChunkWindow],
    speech_intervals: Sequence[tuple[float, float]] = (),
    vad_based_chunks: bool,
    pause_threshold_seconds: float,
) -> dict[str, object]:
    if not timestamps:
        return {
            "status": "not_requested_or_unavailable",
            "timestamp_source": None,
            "granularity": None,
            "unit_count": 0,
            "alignment_text_coverage_ratio": None,
            "reading_segments": [],
            "pause_candidates": [],
            "rate": None,
        }

    granularities = sorted({str(item["granularity"]) for item in timestamps})
    granularity = granularities[0] if len(granularities) == 1 else "mixed"
    aligned_text = "".join(str(item["text"]) for item in timestamps)
    normalized_transcript = _normalize_text(transcript)
    normalized_aligned = _normalize_text(aligned_text)
    coverage_ratio = (
        min(len(normalized_aligned) / len(normalized_transcript), 1.0)
        if normalized_transcript
        else None
    )
    speech_seconds = _union_duration(
        [(_as_float(item["start"]), _as_float(item["end"])) for item in timestamps]
    )
    rate_unit = "chars_per_minute" if granularity == "char" else "words_per_minute"
    count = len(timestamps)
    pauses = _pause_candidates(
        timestamps,
        chunks=chunks,
        speech_intervals=speech_intervals,
        vad_based_chunks=vad_based_chunks,
        threshold=pause_threshold_seconds,
    )
    return {
        "status": "aligned" if coverage_ratio is None or coverage_ratio >= 0.98 else "partial",
        "timestamp_source": "qwen3_forced_aligner",
        "granularity": granularity,
        "unit_count": count,
        "alignment_text_coverage_ratio": coverage_ratio,
        "reading_segments": _reading_segments(timestamps, granularity=granularity),
        "pause_candidates": pauses,
        "pause_method": (
            "alignment gaps; candidates spanning VAD-separated chunks carry alignment_and_vad evidence; "
            "candidates are not semantic pause labels"
        ),
        "rate": {
            "unit": rate_unit,
            "count": count,
            "elapsed_seconds": duration,
            "aligned_speech_seconds": speech_seconds,
            "including_pauses": count * 60 / duration if duration > 0 else None,
            "aligned_speech_only": count * 60 / speech_seconds if speech_seconds > 0 else None,
        },
    }


def _reading_segments(timestamps: list[dict[str, object]], *, granularity: str) -> list[dict[str, object]]:
    groups: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for item in timestamps:
        if current:
            gap = _as_float(item["start"]) - _as_float(current[-1]["end"])
            span = _as_float(item["end"]) - _as_float(current[0]["start"])
            if gap >= 0.8 or span > 15.0 or len(current) >= 40:
                groups.append(current)
                current = []
        current.append(item)
    if current:
        groups.append(current)
    separator = "" if granularity == "char" else " "
    return [
        {
            "start": _as_float(group[0]["start"]),
            "end": _as_float(group[-1]["end"]),
            "text": separator.join(str(item["text"]) for item in group),
            "organization": "alignment_gap_group",
            "semantic_sentence": False,
        }
        for group in groups
    ]


def _pause_candidates(
    timestamps: list[dict[str, object]],
    *,
    chunks: Sequence[ChunkWindow],
    speech_intervals: Sequence[tuple[float, float]],
    vad_based_chunks: bool,
    threshold: float,
) -> list[dict[str, object]]:
    pauses: list[dict[str, object]] = []
    for previous, current in zip(timestamps, timestamps[1:]):
        start = _as_float(previous["end"])
        end = _as_float(current["start"])
        if end - start < threshold:
            continue
        speech_overlap = _union_duration(
            [
                (max(speech_start, start), min(speech_end, end))
                for speech_start, speech_end in speech_intervals
                if speech_start < end and speech_end > start
            ]
        )
        joint_vad = (
            vad_based_chunks
            and bool(speech_intervals)
            and speech_overlap <= (end - start) * 0.25
        )
        pauses.append(
            {
                "start": start,
                "end": end,
                "duration": end - start,
                "evidence": "alignment_and_vad" if joint_vad else "alignment_gap",
            }
        )
    return pauses


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    current_start: float | None = None
    current_end: float | None = None
    for start, end in sorted(intervals):
        if current_start is None or current_end is None:
            current_start, current_end = start, end
        elif start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    if current_start is not None and current_end is not None:
        total += current_end - current_start
    return total


def _normalize_text(text: str) -> str:
    return _PUNCTUATION.sub("", text).casefold()


def _as_float(value: object) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"expected numeric value, got {type(value).__name__}")
