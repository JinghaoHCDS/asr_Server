from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from asr_server.adapters.base import TranscriptionResult, TranscriptionTimings
from asr_server.audio.transcript import TranscriptDocument, build_transcript_document


class ChunkLike(Protocol):
    @property
    def index(self) -> int: ...

    @property
    def start(self) -> float: ...

    @property
    def end(self) -> float: ...

    @property
    def duration(self) -> float: ...


@dataclass(frozen=True)
class MergedTranscription:
    text: str
    language: str
    duration: float
    warnings: list[str]
    timings: TranscriptionTimings
    chunks: list[dict[str, object]]
    segments: list[dict[str, object]]
    timestamps: list[dict[str, object]]


def merge_transcription_results(
    chunks: Sequence[ChunkLike],
    results: list[TranscriptionResult],
    *,
    source_duration: float,
    preserve_segments: bool,
    timings: TranscriptionTimings,
    speaker_scope: Literal["global", "chunk", "mixed"] = "chunk",
) -> MergedTranscription:
    raw_segments = [
        {
            "start": chunk.start,
            "end": chunk.end,
            "text": result.text,
            "language": result.language,
        }
        for chunk, result in zip(chunks, results, strict=True)
    ]
    document = build_transcript_document(
        raw_segments,
        metadata={"source_duration": source_duration},
        timestamp_source="vad_chunk_window",
    )
    segments = _merged_segments(chunks, results, speaker_scope=speaker_scope)
    timestamps = _merged_timestamps(chunks, results)
    warnings = _unique_warnings(results)
    if (
        len(chunks) > 1
        and segments
        and speaker_scope == "chunk"
        and "moss_speaker_labels_are_chunk_local" not in warnings
    ):
        warnings = [*warnings, "moss_speaker_labels_are_chunk_local"]
    language = next((result.language for result in results if result.language), "auto")
    merged_text = document.text
    if segments and all(result.segments for result in results):
        merged_text = "\n".join(str(segment["text"]) for segment in segments if str(segment["text"]).strip())
    return MergedTranscription(
        text=merged_text,
        language=language,
        duration=source_duration,
        warnings=warnings,
        timings=timings,
        chunks=_chunk_payload(chunks, results, document) if preserve_segments else [],
        segments=segments,
        timestamps=timestamps,
    )


def _unique_warnings(results: list[TranscriptionResult]) -> list[str]:
    warnings: list[str] = []
    seen: set[str] = set()
    for result in results:
        for warning in result.warnings:
            if warning not in seen:
                warnings.append(warning)
                seen.add(warning)
    return warnings


def _chunk_payload(
    chunks: Sequence[ChunkLike],
    results: list[TranscriptionResult],
    document: TranscriptDocument,
) -> list[dict[str, object]]:
    payload = []
    for chunk, result, segment in zip(chunks, results, document.segments, strict=True):
        payload.append(
            {
                "index": chunk.index,
                "start": chunk.start,
                "end": chunk.end,
                "duration": chunk.duration,
                "text": segment.text,
                "raw_text": result.text,
                "language": result.language,
                "timestamp_source": segment.timestamp_source,
                "overlap_seconds": segment.overlap_seconds,
                "deduped_prefix_chars": segment.deduped_prefix_chars,
                "warnings": result.warnings,
                "timings": result.timings.to_api(),
                "segments": [
                    {
                        "start": chunk.start + item.start,
                        "end": chunk.start + item.end,
                        "speaker": item.speaker,
                        "text": item.text,
                        **({"source_speaker": item.source_speaker} if item.source_speaker is not None else {}),
                        **(
                            {"speaker_resolution": item.speaker_resolution}
                            if item.speaker_resolution is not None
                            else {}
                        ),
                    }
                    for item in result.segments
                ],
                "timestamps": [
                    {
                        **item.to_api(),
                        "start": chunk.start + item.start,
                        "end": chunk.start + item.end,
                    }
                    for item in result.timestamps
                ],
            }
        )
    return payload


def _merged_segments(
    chunks: Sequence[ChunkLike],
    results: list[TranscriptionResult],
    *,
    speaker_scope: Literal["global", "chunk", "mixed"],
) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    for chunk_position, (chunk, result) in enumerate(zip(chunks, results, strict=True)):
        ownership_start = chunk.start
        ownership_end = chunk.end
        if chunk_position > 0:
            previous = chunks[chunk_position - 1]
            if previous.end > chunk.start:
                ownership_start = (previous.end + chunk.start) / 2
        if chunk_position + 1 < len(chunks):
            following = chunks[chunk_position + 1]
            if following.start < chunk.end:
                ownership_end = (following.start + chunk.end) / 2
        for segment in result.segments:
            absolute_start = chunk.start + segment.start
            absolute_end = chunk.start + segment.end
            midpoint = (absolute_start + absolute_end) / 2
            if midpoint < ownership_start or midpoint > ownership_end:
                continue
            speaker = segment.speaker
            scoped_speaker = (
                speaker
                if speaker_scope != "chunk"
                else f"chunk-{chunk.index:04d}:{speaker}" if speaker is not None else None
            )
            segment_scope = (
                "global" if speaker_scope == "mixed" and speaker is not None else speaker_scope
            )
            payload: dict[str, object] = {
                "start": absolute_start,
                "end": absolute_end,
                "speaker": scoped_speaker,
                "speaker_label": speaker,
                "speaker_scope": segment_scope,
                "chunk_index": chunk.index,
                "text": segment.text,
            }
            if segment.source_speaker is not None:
                payload["source_speaker"] = segment.source_speaker
            if segment.speaker_resolution is not None:
                payload["speaker_resolution"] = segment.speaker_resolution
            segments.append(payload)
    return segments


def _merged_timestamps(
    chunks: Sequence[ChunkLike],
    results: list[TranscriptionResult],
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for chunk_position, (chunk, result) in enumerate(zip(chunks, results, strict=True)):
        for timestamp_item in result.timestamps:
            absolute_start = chunk.start + timestamp_item.start
            absolute_end = chunk.start + timestamp_item.end
            items.append(
                {
                    **timestamp_item.to_api(),
                    "start": absolute_start,
                    "end": absolute_end,
                    "chunk_index": chunk.index,
                    "_chunk_position": chunk_position,
                }
            )
    items.sort(key=lambda payload: (_as_float(payload["start"]), _as_float(payload["end"])))
    deduped: list[dict[str, object]] = []
    for payload_item in items:
        duplicate_index = _overlap_duplicate_index(payload_item, deduped, chunks)
        if duplicate_index is None:
            deduped.append(payload_item)
            continue
        existing = deduped[duplicate_index]
        left_position = min(_as_int(existing["_chunk_position"]), _as_int(payload_item["_chunk_position"]))
        right_position = max(_as_int(existing["_chunk_position"]), _as_int(payload_item["_chunk_position"]))
        boundary = (chunks[left_position].end + chunks[right_position].start) / 2
        existing_midpoint = (_as_float(existing["start"]) + _as_float(existing["end"])) / 2
        item_midpoint = (_as_float(payload_item["start"]) + _as_float(payload_item["end"])) / 2
        if abs(item_midpoint - boundary) < abs(existing_midpoint - boundary):
            deduped[duplicate_index] = payload_item
    for payload_item in deduped:
        payload_item.pop("_chunk_position", None)
    deduped.sort(key=lambda payload: (_as_float(payload["start"]), _as_float(payload["end"])))
    return deduped


def _overlap_duplicate_index(
    candidate: dict[str, object],
    existing_items: list[dict[str, object]],
    chunks: Sequence[ChunkLike],
) -> int | None:
    candidate_position = _as_int(candidate["_chunk_position"])
    candidate_midpoint = (_as_float(candidate["start"]) + _as_float(candidate["end"])) / 2
    for index in range(len(existing_items) - 1, -1, -1):
        existing = existing_items[index]
        existing_position = _as_int(existing["_chunk_position"])
        if abs(candidate_position - existing_position) != 1:
            continue
        if str(existing["text"]).casefold() != str(candidate["text"]).casefold():
            continue
        left_position = min(candidate_position, existing_position)
        right_position = max(candidate_position, existing_position)
        overlap_start = max(chunks[left_position].start, chunks[right_position].start)
        overlap_end = min(chunks[left_position].end, chunks[right_position].end)
        existing_midpoint = (_as_float(existing["start"]) + _as_float(existing["end"])) / 2
        if overlap_start <= candidate_midpoint <= overlap_end and overlap_start <= existing_midpoint <= overlap_end:
            return index
    return None


def _as_float(value: object) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"expected numeric value, got {type(value).__name__}")


def _as_int(value: object) -> int:
    if isinstance(value, int | str):
        return int(value)
    raise TypeError(f"expected integer value, got {type(value).__name__}")
