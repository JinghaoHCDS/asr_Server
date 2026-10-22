from __future__ import annotations

import io
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from asr_server.adapters.base import AudioInput, AudioPath, GenerationMetrics, TimestampItem, TranscriptionResult
from asr_server.adapters.mock import MockAsrAdapter
from asr_server.adapters.qwen import _QwenWorkerBackend
from asr_server.config import Settings
from asr_server.errors import AsrError
from asr_server.lifecycle import ModelLifecycleManager
from asr_server.quality.checkpoints import Checkpoints
from asr_server.quality.glossary import glossary_context, normalize_terms
from asr_server.quality.pipeline import _owned_text
from asr_server.quality.splitter import QualityChunk, plan_quality_chunks
from asr_server.quality.segments import reading_segments
from asr_server.registry import default_models
from asr_server.transcription import TranscriptionRequest, run_transcription_path, validate_transcription_request


def wav_bytes(seconds: float, amplitude: int = 0) -> bytes:
    target = io.BytesIO()
    with wave.open(target, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(amplitude.to_bytes(2, "little", signed=True) * int(seconds * 16000))
    return target.getvalue()


def request(**changes: object) -> TranscriptionRequest:
    original = TranscriptionRequest(
        model="qwen3-asr-1.7b", language="auto", response_format="verbose_json", timestamps="auto",
        backend="auto", max_new_tokens=None, context="", hotwords=None, split_strategy="auto",
        max_chunk_seconds=None, overlap_seconds=None, preserve_segments=True, quality_profile="quality_v1",
    )
    return replace(original, **changes)  # type: ignore[arg-type]


def test_low_energy_planner_preserves_all_samples_and_bounded_context(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes(370))
    chunks, duration = plan_quality_chunks(source, target=180, padding=1)
    assert duration == 370
    assert chunks[0].core_start == 0
    assert chunks[-1].core_end == duration
    assert all(left.core_end == right.core_start for left, right in zip(chunks, chunks[1:]))
    assert all(chunk.duration <= 180 for chunk in chunks)
    assert sum(chunk.core_end - chunk.core_start for chunk in chunks) == duration


def test_planner_prefers_quiet_boundary_without_discarding_quiet_audio(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"\x10\x10" * (27 * 16000))
        target.writeframes(b"\x00\x00" * 16000)
        target.writeframes(b"\x10\x10" * (32 * 16000))
    chunks, duration = plan_quality_chunks(source, target=30)
    assert chunks[0].core_end == pytest.approx(27.9)
    assert duration == 60
    assert chunks[1].core_start == chunks[0].core_end


def test_alias_normalization_is_non_cascading_and_rejects_collisions() -> None:
    snapshot: dict[str, object] = {"entries": [
        {"canonical": "Qwen3-ASR", "aliases": ["千问 ASR"], "auto_normalize": True},
        {"canonical": "Other", "aliases": ["千问 ASR"], "auto_normalize": True},
        {"canonical": "ASR", "aliases": ["语音识别"], "auto_normalize": True},
        {"canonical": "Automatic Speech Recognition", "aliases": ["ASR"], "auto_normalize": True},
    ]}
    normalized, changes, unresolved = normalize_terms("千问 ASR，语音识别", snapshot)
    assert normalized == "千问 ASR，ASR"
    assert len(changes) == 1
    assert unresolved == ["千问 ASR"]


def test_latin_alias_respects_word_boundaries() -> None:
    snapshot: dict[str, object] = {"entries": [{"canonical": "Qwen", "aliases": ["qwen"], "auto_normalize": True}]}
    normalized, _changes, _unresolved = normalize_terms("qwen qwenish", snapshot)
    assert normalized == "Qwen qwenish"


def test_user_hotwords_precede_snapshot_and_omissions_are_reported() -> None:
    snapshot: dict[str, object] = {"version": 2, "entries": [{"canonical": f"word-{i}"} for i in range(110)]}
    prompt, report = glossary_context("", "personal", snapshot)
    assert "personal" in prompt
    used = report["used"]
    assert isinstance(used, list) and used[0] == "personal" and len(used) == 100
    assert report["omitted"]


def test_glossary_reporting_matches_casefolded_model_prompt() -> None:
    snapshot: dict[str, object] = {"entries": [{"canonical": "qwen"}, {"canonical": "Other"}]}
    _prompt, report = glossary_context("", "Qwen", snapshot)
    assert report["used"] == ["Qwen", "Other"]
    assert report["omitted"] == []
    summary = report["context"]
    assert isinstance(summary, dict) and summary["hotword_count"] == 2


def test_timestamp_ownership_keeps_real_repeated_words_and_punctuation() -> None:
    result = TranscriptionResult("你好，你好。再见！", 4, "zh", [], timestamps=[
        TimestampItem("你好", 0, 1, "word"), TimestampItem("你好", 1, 2, "word"), TimestampItem("再见", 2, 3, "word"),
    ])
    text, items = _owned_text(result, QualityChunk(1, 0, 4, 1, 4))
    assert text == "你好。再见！"
    assert [item.text for item in items] == ["你好", "再见"]


def test_checkpoint_mismatch_and_retry_budget_survive_restart(tmp_path: Path) -> None:
    checkpoint = Checkpoints(tmp_path, "identity")
    checkpoint.reserve_review(3, 10)
    restored = Checkpoints(tmp_path, "identity")
    assert restored.extra_seconds == 3
    with pytest.raises(AsrError) as error:
        restored.reserve_review(8, 10)
    assert error.value.code == "review_budget_exhausted"
    with pytest.raises(AsrError) as error:
        Checkpoints(tmp_path, "another-model-revision")
    assert error.value.code == "checkpoint_version_mismatch"


class RecordingAdapter(MockAsrAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[int | None] = []
        self.aligned: list[str] = []

    async def transcribe(self, audio: AudioInput, *, model_id: str, backend: str, language: str, context: str, timestamps: str = "none", max_new_tokens: int | None = None) -> TranscriptionResult:
        del model_id, backend, language, context, timestamps
        self.calls.append(max_new_tokens)
        return TranscriptionResult("千问 ASR。", 3, "zh", [], timestamps=[TimestampItem("千问", 0, 1, "word"), TimestampItem("ASR", 1, 2, "word")], generation=GenerationMetrics(max_new_tokens=max_new_tokens))

    async def align_text(self, audio: AudioInput, *, text: str, language: str = "auto", timestamps: str = "auto") -> list[TimestampItem]:
        del audio, language, timestamps
        self.aligned.append(text)
        return [TimestampItem("Qwen3-ASR", 0, 2, "word")]


@pytest.mark.asyncio
async def test_quality_normalization_realigns_and_resumes_completed_chunks(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes(3))
    adapter = RecordingAdapter()
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: adapter, idle_unload_seconds=0)
    snapshot: dict[str, object] = {"version": 1, "entries": [{"canonical": "Qwen3-ASR", "aliases": ["千问 ASR"], "auto_normalize": True}]}
    parameters = request(glossary_snapshot=snapshot)
    try:
        first = await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=parameters, checkpoint_dir=tmp_path / "checkpoints")
        second = await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=parameters, checkpoint_dir=tmp_path / "checkpoints")
        assert first["text"] == second["text"] == "Qwen3-ASR。"
        assert first["raw_text"] == "千问 ASR。"
        assert adapter.calls == [4096]
        assert adapter.aligned == ["Qwen3-ASR。", "Qwen3-ASR。"]
        assert first["timestamps"] == [TimestampItem("Qwen3-ASR", 0, 2, "word").to_api()]
        assert isinstance(first["quality"], dict) and first["quality"]["validated"] is False
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_quality_respects_explicit_generation_limit(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes(3))
    adapter = RecordingAdapter()
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: adapter, idle_unload_seconds=0)
    try:
        await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request(max_new_tokens=123))
        assert adapter.calls == [123]
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_unrecoverable_missing_audio_never_reports_completed(tmp_path: Path) -> None:
    class EmptyAdapter(RecordingAdapter):
        async def transcribe(self, audio: AudioInput, *, model_id: str, backend: str, language: str, context: str, timestamps: str = "none", max_new_tokens: int | None = None) -> TranscriptionResult:
            raise AsrError(422, "generation_truncated", "truncated")

    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes(3, amplitude=100))
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: EmptyAdapter(), idle_unload_seconds=0)
    try:
        with pytest.raises(AsrError) as error:
            await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request())
        assert error.value.code == "quality_incomplete"
        assert error.value.details["end"] == 3
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_truncation_is_resegmented_within_real_audio_budget(tmp_path: Path) -> None:
    class OnceTruncated(RecordingAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.failed_once = False

        async def transcribe(self, audio: AudioInput, *, model_id: str, backend: str, language: str, context: str, timestamps: str = "none", max_new_tokens: int | None = None) -> TranscriptionResult:
            if not self.failed_once:
                self.failed_once = True
                raise AsrError(422, "generation_truncated", "truncated first input")
            return await super().transcribe(audio, model_id=model_id, backend=backend, language=language, context=context, timestamps=timestamps, max_new_tokens=max_new_tokens)

    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes(240))
    adapter = OnceTruncated()
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: adapter, idle_unload_seconds=0)
    try:
        result = await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request(), checkpoint_dir=tmp_path / "checkpoints")
        quality = result["quality"]
        assert isinstance(quality, dict)
        assert quality["review_audio_seconds"] == 60
        assert quality["review_budget_seconds"] == 72
        assert len(adapter.calls) == 5  # Two repaired half-windows plus three untouched chunks.
        assert str(result["text"]).count("千问 ASR") == 5
        restored = Checkpoints(tmp_path / "checkpoints", str(quality["fingerprint"]))
        assert restored.review_attempts["chunk:0"] == 1
    finally:
        await manager.shutdown()


def test_quality_cannot_be_used_for_moss_or_unaligned_deployments() -> None:
    manager = ModelLifecycleManager(default_models(enable_moss=True), lambda _: MockAsrAdapter())
    with pytest.raises(AsrError) as error:
        validate_transcription_request(manager, request(model="moss-transcribe-diarize-0.9b"))
    assert error.value.code == "capability_not_supported"
    with pytest.raises(AsrError) as error:
        validate_transcription_request(manager, request())
    assert error.value.code == "capability_not_supported"


def test_reading_segments_preserve_punctuation_and_original_aliases() -> None:
    chunks: list[dict[str, object]] = [{
        "index": 0, "start": 0, "end": 12, "text": "你好。Qwen3-ASR 很好！再见。", "raw_text": "你好。千问 ASR 很好！再见。",
        "timestamps": [
            TimestampItem("你好", 0, 2, "word").to_api(),
            TimestampItem("Qwen3-ASR", 3, 4, "word").to_api(),
            TimestampItem("很好", 4, 5, "word").to_api(),
            TimestampItem("再见", 9, 10, "word").to_api(),
        ],
    }]
    segments = reading_segments(chunks)
    assert [item["text"] for item in segments] == ["你好。", "Qwen3-ASR 很好！", "再见。"]
    assert segments[1]["raw_text"] == "千问 ASR 很好！"
    assert all(float(str(item["end"])) - float(str(item["start"])) <= 6 for item in segments)


@pytest.mark.asyncio
async def test_qwen_supplied_text_alignment_does_not_transcribe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    backend = _QwenWorkerBackend("qwen3-asr-1.7b", enable_alignment=True)
    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes(1))
    seen: list[str] = []

    def align(path: str, transcript: str, language: str, *, requested_granularity: str) -> tuple[list[TimestampItem], list[str]]:
        del path, language, requested_granularity
        seen.append(transcript)
        return [TimestampItem(transcript, 0, 1, "word")], []

    monkeypatch.setattr(backend, "_align", align)
    aligned = await backend.align_text(AudioPath(source), text="用户修改后的原稿")
    assert seen == ["用户修改后的原稿"]
    assert aligned[0].text == "用户修改后的原稿"


@pytest.mark.parametrize('amplitude', [0, 1, 20])
async def test_quiet_nonzero_audio_is_not_silently_accepted_as_digital_silence(tmp_path: Path, amplitude: int) -> None:
    class EmptyAdapter(RecordingAdapter):
        async def transcribe(self, audio: AudioInput, *, model_id: str, backend: str, language: str, context: str, timestamps: str = "none", max_new_tokens: int | None = None) -> TranscriptionResult:
            raise AsrError(422, "empty_transcription", "no text")
    source = tmp_path / 'quiet.wav'
    source.write_bytes(wav_bytes(1, amplitude=amplitude))
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: EmptyAdapter(), idle_unload_seconds=0)
    try:
        if amplitude:
            with pytest.raises(AsrError) as error:
                await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request())
            assert error.value.code == 'quality_incomplete'
        else:
            result = await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request())
            assert result['text'] == ''
    finally:
        await manager.shutdown()
