from __future__ import annotations

from dataclasses import dataclass

import pytest

from asr_server.adapters.base import TimestampItem, TranscriptionResult, TranscriptionTimings
from asr_server.audio.merger import merge_transcription_results
from asr_server.audio.timing import timing_analysis


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def test_merge_preserves_original_timeline_and_overlap_ownership() -> None:
    chunks = [Chunk(0, 100.0, 110.0), Chunk(1, 108.0, 118.0)]
    results = [
        TranscriptionResult(
            text="甲乙",
            duration=10.0,
            language="zh",
            warnings=[],
            timestamps=[
                TimestampItem("甲", 1.0, 1.2, "char"),
                TimestampItem("乙", 9.2, 9.5, "char"),
            ],
        ),
        TranscriptionResult(
            text="乙丙",
            duration=10.0,
            language="zh",
            warnings=[],
            timestamps=[
                TimestampItem("乙", 0.2, 0.5, "char"),
                TimestampItem("丙", 3.0, 3.4, "char"),
            ],
        ),
    ]

    merged = merge_transcription_results(
        chunks,
        results,
        source_duration=118.0,
        preserve_segments=True,
        timings=TranscriptionTimings(),
    )

    assert [(item["text"], item["start"], item["end"]) for item in merged.timestamps] == [
        ("甲", 101.0, 101.2),
        ("乙", 109.2, 109.5),
        ("丙", 111.0, 111.4),
    ]
    assert all(item["source"] == "qwen3_forced_aligner" for item in merged.timestamps)


def test_timing_analysis_distinguishes_elapsed_and_speech_rates_and_vad_pause() -> None:
    timestamps = [
        TimestampItem("你", 10.0, 10.2, "char").to_api(),
        TimestampItem("好", 12.0, 12.2, "char").to_api(),
    ]
    analysis = timing_analysis(
        timestamps,
        transcript="你好",
        duration=20.0,
        chunks=[Chunk(0, 0.0, 20.0)],
        speech_intervals=[(9.9, 10.3), (11.9, 12.3)],
        vad_based_chunks=True,
        pause_threshold_seconds=0.8,
    )

    assert analysis["status"] == "aligned"
    assert analysis["granularity"] == "char"
    rate = analysis["rate"]
    assert isinstance(rate, dict)
    assert rate["unit"] == "chars_per_minute"
    assert rate["including_pauses"] == pytest.approx(6.0)
    assert rate["aligned_speech_only"] == pytest.approx(300.0)
    assert analysis["pause_candidates"] == [
        {"start": 10.2, "end": 12.0, "duration": 1.8000000000000007, "evidence": "alignment_and_vad"}
    ]


def test_missing_alignment_never_fabricates_timestamps() -> None:
    analysis = timing_analysis(
        [],
        transcript="has text",
        duration=3.0,
        chunks=[Chunk(0, 0.0, 3.0)],
        vad_based_chunks=False,
        pause_threshold_seconds=0.8,
    )

    assert analysis["status"] == "not_requested_or_unavailable"
    assert analysis["reading_segments"] == []
    assert analysis["rate"] is None
