from __future__ import annotations

import wave
from array import array
from pathlib import Path

import pytest

from asr_server.adapters.base import (
    AudioComposition,
    AudioPath,
    AudioSilence,
    TranscriptionResult,
    TranscriptionSegment,
)
from asr_server.audio.splitter import ChunkDescriptor
from asr_server.diarization.anchor_replay import AnchorReplayConfig, AnchorReplaySequence
from asr_server.workers.audio import materialized_audio_path


def _write_wav(path: Path, duration: float, sample_rate: int = 100) -> None:
    samples = array("h", [1_000] * int(duration * sample_rate))
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(samples.tobytes())


def _result(*segments: tuple[float, float, str, str], duration: float = 100.0) -> TranscriptionResult:
    return TranscriptionResult(
        text=" ".join(item[3] for item in segments),
        duration=duration,
        language="zh",
        warnings=[],
        segments=[TranscriptionSegment(start, end, text, speaker) for start, end, speaker, text in segments],
    )


def test_anchor_replay_keeps_four_speakers_stable_when_local_labels_are_permuted(tmp_path: Path) -> None:
    audio_path = tmp_path / "source.wav"
    _write_wav(audio_path, 200.0)
    chunks = [
        ChunkDescriptor(index=0, start=0.0, end=100.0, audio_path=audio_path),
        ChunkDescriptor(index=1, start=100.0, end=200.0, audio_path=audio_path),
    ]
    sequence = AnchorReplaySequence(
        chunks,
        config=AnchorReplayConfig(anchor_seconds=5.0, min_anchor_seconds=2.0, silence_seconds=0.5),
    )

    first = sequence.accept(
        0,
        _result(
            (1.0, 7.0, "S01", "甲"),
            (10.0, 16.0, "S02", "乙"),
            (20.0, 26.0, "S03", "丙"),
            (30.0, 36.0, "S04", "丁"),
        ),
    )
    assert [item.speaker for item in first.segments] == ["speaker-0001", "speaker-0002", "speaker-0003", "speaker-0004"]

    composed = sequence.audio_for(1)
    assert isinstance(composed, AudioComposition)
    assert isinstance(composed.parts[-1], AudioPath)
    prefix = composed.prefix_duration
    assert prefix == pytest.approx(22.0)

    second = sequence.accept(
        1,
        _result(
            (0.0, 5.0, "S04", "甲锚"),
            (5.5, 10.5, "S02", "乙锚"),
            (11.0, 16.0, "S01", "丙锚"),
            (16.5, 21.5, "S03", "丁锚"),
            (prefix + 1.0, prefix + 3.0, "S03", "丁继续"),
            (prefix + 4.0, prefix + 6.0, "S04", "甲继续"),
            duration=122.0,
        ),
    )

    assert [(item.start, item.speaker, item.text) for item in second.segments] == [
        (1.0, "speaker-0004", "丁继续"),
        (4.0, "speaker-0001", "甲继续"),
    ]
    summary = sequence.summary()
    assert summary["status"] == "complete"
    assert summary["speaker_count"] == 4
    assert summary["unresolved_segments"] == 0


def test_anchor_replay_does_not_force_a_fifth_speaker_into_a_to_d_scope(tmp_path: Path) -> None:
    audio_path = tmp_path / "source.wav"
    _write_wav(audio_path, 100.0)
    sequence = AnchorReplaySequence(
        [ChunkDescriptor(index=0, start=0.0, end=100.0, audio_path=audio_path)],
        config=AnchorReplayConfig(min_anchor_seconds=1.0, max_speakers=4),
    )

    result = sequence.accept(
        0,
        _result(
            (0.0, 2.0, "S01", "一"),
            (2.0, 4.0, "S02", "二"),
            (4.0, 6.0, "S03", "三"),
            (6.0, 8.0, "S04", "四"),
            (8.0, 10.0, "S05", "五"),
        ),
    )

    assert [segment.speaker for segment in result.segments] == [
        "speaker-0001",
        "speaker-0002",
        "speaker-0003",
        "speaker-0004",
        None,
    ]
    assert result.segments[-1].speaker_resolution == "unresolved"
    assert sequence.summary()["speaker_capacity_exceeded"] == 1


def test_anchor_replay_does_not_merge_two_global_speakers_when_anchor_labels_conflict(tmp_path: Path) -> None:
    audio_path = tmp_path / "source.wav"
    _write_wav(audio_path, 40.0)
    chunks = [
        ChunkDescriptor(index=0, start=0.0, end=20.0, audio_path=audio_path),
        ChunkDescriptor(index=1, start=20.0, end=40.0, audio_path=audio_path),
    ]
    sequence = AnchorReplaySequence(
        chunks,
        config=AnchorReplayConfig(anchor_seconds=4.0, min_anchor_seconds=2.0, silence_seconds=0.5),
    )
    sequence.accept(0, _result((0.0, 5.0, "S01", "甲"), (8.0, 13.0, "S02", "乙"), duration=20.0))
    composed = sequence.audio_for(1)
    assert isinstance(composed, AudioComposition)
    prefix = composed.prefix_duration

    result = sequence.accept(
        1,
        _result(
            (0.0, 4.0, "S01", "甲锚"),
            (4.5, 8.5, "S01", "乙锚冲突"),
            (prefix + 1.0, prefix + 3.0, "S01", "不能安全归属"),
            duration=prefix + 20.0,
        ),
    )

    assert result.segments[0].speaker is None
    assert sequence.summary()["status"] == "partial"
    assert sequence.summary()["conflicts"] == 1
    assert sequence.summary()["unresolved_segments"] == 1


def test_anchor_replay_does_not_promote_a_short_intro_voice_to_global_identity(tmp_path: Path) -> None:
    audio_path = tmp_path / "source.wav"
    _write_wav(audio_path, 40.0)
    chunks = [
        ChunkDescriptor(index=0, start=0.0, end=20.0, audio_path=audio_path),
        ChunkDescriptor(index=1, start=20.0, end=40.0, audio_path=audio_path),
    ]
    sequence = AnchorReplaySequence(chunks)

    first = sequence.accept(
        0,
        _result(
            (0.0, 4.0, "HOST_A", "主谈人甲"),
            (5.0, 9.0, "HOST_B", "主谈人乙"),
            (10.0, 11.1, "INTRO", "片头短句一"),
            (11.2, 12.3, "INTRO", "片头短句二"),
            duration=20.0,
        ),
    )

    assert [item.speaker for item in first.segments] == ["speaker-0001", "speaker-0002", None, None]
    assert [item.speaker_resolution for item in first.segments] == ["initial", "initial", "unresolved", "unresolved"]
    replay = sequence.audio_for(1)
    assert isinstance(replay, AudioComposition)
    assert len([part for part in replay.parts if isinstance(part, AudioPath)]) == 3
    assert sequence.summary()["speaker_count"] == 2
    assert sequence.summary()["unresolved_segments"] == 2


def test_anchor_replay_reconnects_a_speaker_after_an_absent_chunk(tmp_path: Path) -> None:
    audio_path = tmp_path / "source.wav"
    _write_wav(audio_path, 60.0)
    chunks = [
        ChunkDescriptor(index=index, start=index * 20.0, end=(index + 1) * 20.0, audio_path=audio_path)
        for index in range(3)
    ]
    sequence = AnchorReplaySequence(
        chunks,
        config=AnchorReplayConfig(anchor_seconds=3.0, min_anchor_seconds=2.0, silence_seconds=0.5),
    )
    sequence.accept(0, _result((0.0, 4.0, "S01", "甲"), (8.0, 12.0, "S02", "乙"), duration=20.0))

    second_input = sequence.audio_for(1)
    assert isinstance(second_input, AudioComposition)
    prefix = second_input.prefix_duration
    sequence.accept(
        1,
        _result(
            (0.0, 3.0, "A", "甲锚"),
            (3.5, 6.5, "B", "乙锚"),
            (prefix + 1.0, prefix + 3.0, "A", "只有甲"),
            duration=prefix + 20.0,
        ),
    )

    third_input = sequence.audio_for(2)
    assert isinstance(third_input, AudioComposition)
    prefix = third_input.prefix_duration
    third = sequence.accept(
        2,
        _result(
            (0.0, 3.0, "X", "甲锚"),
            (3.5, 6.5, "Y", "乙锚"),
            (prefix + 1.0, prefix + 3.0, "Y", "乙回来了"),
            duration=prefix + 20.0,
        ),
    )

    assert third.segments[0].speaker == "speaker-0002"
    assert sequence.summary()["status"] == "complete"


def test_late_speaker_stays_candidate_until_replayed_in_a_later_chunk(tmp_path: Path) -> None:
    audio_path = tmp_path / "source.wav"
    _write_wav(audio_path, 60.0)
    chunks = [
        ChunkDescriptor(index=index, start=index * 20.0, end=(index + 1) * 20.0, audio_path=audio_path)
        for index in range(3)
    ]
    sequence = AnchorReplaySequence(
        chunks,
        config=AnchorReplayConfig(anchor_seconds=3.0, min_anchor_seconds=2.0, silence_seconds=0.5),
    )
    sequence.accept(0, _result((0.0, 4.0, "A", "甲"), (6.0, 10.0, "B", "乙"), duration=20.0))
    second_input = sequence.audio_for(1)
    assert isinstance(second_input, AudioComposition)
    prefix = second_input.prefix_duration
    second = sequence.accept(
        1,
        _result(
            (0.0, 3.0, "X", "甲锚"),
            (3.5, 6.5, "Y", "乙锚"),
            (prefix + 1.0, prefix + 4.0, "Z", "新人"),
            duration=prefix + 20.0,
        ),
    )
    assert second.segments[0].speaker == "speaker-0003"
    assert second.segments[0].speaker_resolution == "new_candidate"
    assert sequence.summary()["status"] == "partial"
    assert sequence.summary()["candidate_speakers"] == 1

    third_input = sequence.audio_for(2)
    assert isinstance(third_input, AudioComposition)
    prefix = third_input.prefix_duration
    third = sequence.accept(
        2,
        _result(
            (0.0, 3.0, "L1", "甲锚"),
            (3.5, 6.5, "L2", "乙锚"),
            (7.0, 10.0, "L3", "新人锚"),
            (prefix + 1.0, prefix + 3.0, "L3", "新人继续"),
            duration=prefix + 20.0,
        ),
    )
    assert third.segments[0].speaker == "speaker-0003"
    assert third.segments[0].speaker_resolution == "anchor"
    assert sequence.summary()["status"] == "complete"
    assert sequence.summary()["candidate_speakers"] == 0


def test_worker_materializes_anchor_slices_silence_and_body_without_changing_duration(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_wav(source, 10.0)
    composition = AudioComposition(
        (
            AudioPath(source, start=1.0, end=3.0),
            AudioSilence(1.0),
            AudioPath(source, start=5.0, end=6.0),
        ),
        prefix_duration=3.0,
    )

    with materialized_audio_path(composition) as materialized:
        assert materialized.exists()
        with wave.open(str(materialized), "rb") as audio:
            assert audio.getnframes() / audio.getframerate() == pytest.approx(4.0)

    assert not materialized.exists()


def test_one_local_label_conflicting_with_three_anchors_is_never_reaccepted(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_wav(source, 40.0)
    chunks = [
        ChunkDescriptor(index=0, start=0.0, end=20.0, audio_path=source),
        ChunkDescriptor(index=1, start=20.0, end=40.0, audio_path=source),
    ]
    sequence = AnchorReplaySequence(
        chunks,
        config=AnchorReplayConfig(anchor_seconds=3.0, min_anchor_seconds=2.0, silence_seconds=0.5),
    )
    sequence.accept(
        0,
        _result(
            (0.0, 4.0, "A", "甲"),
            (5.0, 9.0, "B", "乙"),
            (10.0, 14.0, "C", "丙"),
            duration=20.0,
        ),
    )
    replay = sequence.audio_for(1)
    assert isinstance(replay, AudioComposition)
    prefix = replay.prefix_duration
    result = sequence.accept(
        1,
        _result(
            (0.0, 3.0, "SAME", "甲锚"),
            (3.5, 6.5, "SAME", "乙锚"),
            (7.0, 10.0, "SAME", "丙锚"),
            (prefix + 1.0, prefix + 3.0, "SAME", "正文"),
            duration=prefix + 20.0,
        ),
    )

    assert result.segments[0].speaker is None
    assert result.segments[0].speaker_resolution == "unresolved"
