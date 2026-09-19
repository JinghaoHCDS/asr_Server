from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, Sequence

from asr_server.adapters.base import (
    AudioComposition,
    AudioInput,
    AudioPath,
    AudioSilence,
    TranscriptionResult,
    TranscriptionSegment,
)


class AnchorChunk(Protocol):
    @property
    def index(self) -> int: ...

    @property
    def start(self) -> float: ...

    @property
    def end(self) -> float: ...

    @property
    def audio_path(self) -> Path: ...


@dataclass(frozen=True)
class AnchorReplayConfig:
    anchor_seconds: float = 8.0
    min_anchor_seconds: float = 2.0
    silence_seconds: float = 0.5
    max_prefix_seconds: float = 60.0
    max_speakers: int = 4


@dataclass(frozen=True)
class _Anchor:
    global_speaker: str
    path: AudioPath


@dataclass(frozen=True)
class _AnchorInterval:
    global_speaker: str
    start: float
    end: float


class AnchorReplaySequence:
    """Adapt chunk inputs and results while keeping identity evidence local to MOSS.

    The class deliberately makes no voice-identity inference itself. It only
    asks MOSS to label known reference clips and the next body in one input,
    then accepts mappings that are unambiguous inside the anchor intervals.
    """

    def __init__(self, chunks: Sequence[AnchorChunk], *, config: AnchorReplayConfig | None = None) -> None:
        if not chunks:
            raise ValueError("anchor replay requires at least one chunk")
        self._chunks = list(chunks)
        self._config = config or AnchorReplayConfig()
        self._anchors: dict[str, _Anchor] = {}
        self._pending_intervals: dict[int, list[_AnchorInterval]] = {}
        self._prefix_durations: dict[int, float] = {}
        self._next_speaker = 1
        self._conflicts = 0
        self._unresolved_segments = 0
        self._budget_limited = False
        self._candidate_speakers: set[str] = set()
        self._missing_segment_chunks = 0
        self._speaker_capacity_exceeded = 0

    def __len__(self) -> int:
        return len(self._chunks)

    def audio_for(self, index: int) -> AudioInput:
        chunk = self._chunks[index]
        body = AudioPath(path=chunk.audio_path, start=chunk.start, end=chunk.end)
        if index == 0 or not self._anchors:
            self._pending_intervals[index] = []
            self._prefix_durations[index] = 0.0
            return body
        parts: list[AudioPath | AudioSilence] = []
        intervals: list[_AnchorInterval] = []
        cursor = 0.0
        for global_speaker, anchor in sorted(self._anchors.items()):
            anchor_duration = anchor.path.duration or 0.0
            required = anchor_duration + self._config.silence_seconds
            if cursor + required > self._config.max_prefix_seconds:
                self._budget_limited = True
                continue
            parts.append(anchor.path)
            intervals.append(_AnchorInterval(global_speaker, cursor, cursor + anchor_duration))
            cursor += anchor_duration
            if self._config.silence_seconds > 0:
                parts.append(AudioSilence(self._config.silence_seconds))
                cursor += self._config.silence_seconds
        parts.append(body)
        self._pending_intervals[index] = intervals
        self._prefix_durations[index] = cursor
        return AudioComposition(tuple(parts), prefix_duration=cursor)

    def accept(self, index: int, result: TranscriptionResult) -> TranscriptionResult:
        chunk = self._chunks[index]
        intervals = self._pending_intervals.get(index, [])
        prefix_duration = self._prefix_durations.get(index, 0.0)
        if index == 0 or not intervals:
            local_to_global: dict[str, str] = {}
            anchors_complete = True
        else:
            local_to_global, anchors_complete = self._resolve_anchor_labels(intervals, result.segments)
        body_segments: list[TranscriptionSegment] = []
        for segment in result.segments:
            if segment.end <= prefix_duration:
                continue
            if segment.start < prefix_duration:
                self._conflicts += 1
                continue
            body_segments.append(segment)
        anchorable_local_speakers = {
            segment.speaker
            for segment in body_segments
            if segment.speaker is not None
            and segment.end - segment.start >= self._config.min_anchor_seconds
        }
        if result.text.strip() and not body_segments:
            self._missing_segment_chunks += 1
        rewritten: list[TranscriptionSegment] = []
        for segment in body_segments:
            body_start = max(segment.start - prefix_duration, 0.0)
            body_end = max(segment.end - prefix_duration, body_start)
            local_speaker = segment.speaker
            global_speaker = local_to_global.get(local_speaker) if local_speaker is not None else None
            resolution = "anchor" if global_speaker is not None and index > 0 else None
            if global_speaker is not None and index == 0:
                resolution = "initial"
            if (
                global_speaker is None
                and local_speaker is not None
                and local_speaker in anchorable_local_speakers
                and (index == 0 or anchors_complete)
            ):
                allocated = self._allocate_speaker()
                if allocated is not None:
                    global_speaker = allocated
                    local_to_global[local_speaker] = global_speaker
                    resolution = "initial" if index == 0 else "new_candidate"
                    if index > 0:
                        self._candidate_speakers.add(global_speaker)
            if global_speaker is None:
                self._unresolved_segments += 1
                resolution = "unresolved"
            rewritten.append(
                replace(
                    segment,
                    start=body_start,
                    end=body_end,
                    speaker=global_speaker,
                    source_speaker=(
                        f"chunk-{chunk.index:04d}:{local_speaker}" if local_speaker is not None else None
                    ),
                    speaker_resolution=resolution,
                )
            )
        self._update_anchors(chunk, rewritten)
        warnings = list(result.warnings)
        if "moss_anchor_replay" not in warnings:
            warnings.append("moss_anchor_replay")
        if (
            (not anchors_complete or self._conflicts > 0 or self._missing_segment_chunks > 0)
            and "moss_anchor_replay_partial" not in warnings
        ):
            warnings.append("moss_anchor_replay_partial")
        generation = result.generation
        if generation.segment_coverage_end_seconds is not None:
            generation = replace(
                generation,
                segment_coverage_end_seconds=max(
                    generation.segment_coverage_end_seconds - prefix_duration,
                    0.0,
                ),
            )
        return replace(
            result,
            text="\n".join(segment.text for segment in rewritten if segment.text.strip()),
            duration=max(result.duration - prefix_duration, 0.0),
            warnings=warnings,
            segments=rewritten,
            generation=generation,
        )

    def summary(self) -> dict[str, object]:
        partial = (
            self._conflicts > 0
            or self._unresolved_segments > 0
            or self._budget_limited
            or bool(self._candidate_speakers)
            or self._missing_segment_chunks > 0
            or self._speaker_capacity_exceeded > 0
        )
        return {
            "method": "moss_anchor_replay",
            "status": "partial" if partial else "complete",
            "speaker_scope": "mixed" if partial else "global",
            "speaker_count": self._next_speaker - 1,
            "unresolved_segments": self._unresolved_segments,
            "conflicts": self._conflicts,
            "anchor_budget_limited": self._budget_limited,
            "candidate_speakers": len(self._candidate_speakers),
            "missing_segment_chunks": self._missing_segment_chunks,
            "max_speakers": self._config.max_speakers,
            "speaker_capacity_exceeded": self._speaker_capacity_exceeded,
        }

    def _resolve_anchor_labels(
        self,
        intervals: list[_AnchorInterval],
        segments: list[TranscriptionSegment],
    ) -> tuple[dict[str, str], bool]:
        proposed: dict[str, str] = {}
        conflicted_labels: set[str] = set()
        complete = True
        for interval in intervals:
            overlaps: dict[str, float] = {}
            for segment in segments:
                if segment.speaker is None:
                    continue
                overlap = max(min(segment.end, interval.end) - max(segment.start, interval.start), 0.0)
                if overlap > 0:
                    overlaps[segment.speaker] = overlaps.get(segment.speaker, 0.0) + overlap
            if not overlaps:
                complete = False
                continue
            local_speaker, overlap = max(overlaps.items(), key=lambda item: item[1])
            if overlap <= (interval.end - interval.start) * 0.5:
                complete = False
                continue
            if local_speaker in conflicted_labels:
                complete = False
                continue
            previous = proposed.get(local_speaker)
            if previous is not None and previous != interval.global_speaker:
                proposed.pop(local_speaker, None)
                conflicted_labels.add(local_speaker)
                complete = False
                self._conflicts += 1
                continue
            proposed[local_speaker] = interval.global_speaker
            self._candidate_speakers.discard(interval.global_speaker)
        if len(proposed) != len(intervals):
            complete = False
        return proposed, complete

    def _allocate_speaker(self) -> str | None:
        if self._next_speaker > self._config.max_speakers:
            self._speaker_capacity_exceeded += 1
            return None
        speaker = f"speaker-{self._next_speaker:04d}"
        self._next_speaker += 1
        return speaker

    def _update_anchors(self, chunk: AnchorChunk, segments: list[TranscriptionSegment]) -> None:
        for segment in segments:
            if segment.speaker is None:
                continue
            duration = segment.end - segment.start
            if duration < self._config.min_anchor_seconds:
                continue
            clipped_duration = min(duration, self._config.anchor_seconds)
            candidate = _Anchor(
                segment.speaker,
                AudioPath(
                    path=chunk.audio_path,
                    start=chunk.start + segment.start,
                    end=chunk.start + segment.start + clipped_duration,
                ),
            )
            current = self._anchors.get(segment.speaker)
            if current is None or (current.path.duration or 0.0) < clipped_duration:
                self._anchors[segment.speaker] = candidate
