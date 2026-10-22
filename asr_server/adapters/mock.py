from __future__ import annotations

import asyncio
import hashlib
from time import perf_counter

from asr_server.adapters.base import (
    AudioInput,
    AudioComposition,
    AudioPath,
    ContextApplication,
    GenerationMetrics,
    TimestampItem,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionTimings,
)
from asr_server.workers.audio import audio_duration_seconds


MOSS_MOCK_MODEL_ID = "moss-transcribe-diarize-0.9b"


class MockAsrAdapter:
    def __init__(self, delay_seconds: float = 0.0) -> None:
        self.delay_seconds = delay_seconds
        self.loaded_backend: str | None = None

    async def load(self, backend: str, device: str, dtype: str, max_new_tokens: int | None = None) -> None:
        del device, dtype, max_new_tokens
        self.loaded_backend = backend

    async def unload(self, cuda_empty_cache: bool) -> None:
        del cuda_empty_cache
        self.loaded_backend = None

    async def abort(self) -> None:
        self.loaded_backend = None

    async def align_text(
        self, audio: AudioInput, *, text: str, language: str = "auto", timestamps: str = "auto",
    ) -> list[TimestampItem]:
        del language, timestamps
        if not text.strip():
            return []
        # One explicit mock item exercises edit/export plumbing. It does not
        # impersonate word/character alignment and must never be quality proof.
        return [TimestampItem(text, 0.0, audio_duration_seconds(audio), "segment", source="mock", status="mock")]

    async def transcribe(
        self,
        audio: AudioInput,
        *,
        model_id: str,
        backend: str,
        language: str,
        context: str,
        timestamps: str = "none",
        max_new_tokens: int | None = None,
    ) -> TranscriptionResult:
        inference_started = perf_counter()
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        inference_ms = (perf_counter() - inference_started) * 1000
        return self._result(
            audio,
            model_id=model_id,
            backend=backend,
            language=language,
            context=context,
            timestamps=timestamps,
            max_new_tokens=max_new_tokens,
            inference_ms=inference_ms,
            batch=False,
            label=_audio_label(audio),
        )

    async def transcribe_batch(
        self,
        audio_chunks: list[AudioInput],
        *,
        model_id: str,
        backend: str,
        language: str,
        context: str,
        timestamps: str = "none",
        max_new_tokens: int | None = None,
    ) -> list[TranscriptionResult]:
        inference_started = perf_counter()
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        inference_ms = (perf_counter() - inference_started) * 1000
        per_chunk_inference_ms = inference_ms / len(audio_chunks) if audio_chunks else 0.0
        return [
            self._result(
                audio,
                model_id=model_id,
                backend=backend,
                language=language,
                context=context,
                timestamps=timestamps,
                max_new_tokens=max_new_tokens,
                inference_ms=per_chunk_inference_ms,
                batch=True,
                label=_audio_label(audio),
            )
            for audio in audio_chunks
        ]

    def _result(
        self,
        audio: AudioInput,
        *,
        model_id: str,
        backend: str,
        language: str,
        context: str,
        timestamps: str,
        max_new_tokens: int | None,
        inference_ms: float,
        batch: bool,
        label: str,
    ) -> TranscriptionResult:
        del backend, timestamps
        audio_size = (
            audio.path.stat().st_size
            if isinstance(audio, AudioPath)
            else int(audio.duration * 32_000)
            if isinstance(audio, AudioComposition)
            else len(audio)
        )
        text = f"{label}:{audio_size}"
        warnings = ["mock_adapter"]
        if batch:
            warnings.append("mock_batch_adapter")
        if context:
            context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()[:12]
            warnings.append(f"context_received:{len(context)}:{context_hash}")
        if max_new_tokens is not None:
            warnings.append(f"max_new_tokens_received:{max_new_tokens}")
        segments = []
        duration = audio_duration_seconds(audio)
        if model_id == MOSS_MOCK_MODEL_ID:
            segments = [
                TranscriptionSegment(
                    start=0.0,
                    end=min(duration, 0.5),
                    speaker="S01",
                    text=f"{label}:{audio_size}",
                )
            ]
        return TranscriptionResult(
            text=text,
            duration=duration,
            language="zh" if language == "auto" else language,
            warnings=warnings,
            segments=segments,
            timings=TranscriptionTimings(inference_ms=inference_ms),
            generation=GenerationMetrics(max_new_tokens=max_new_tokens),
            context=ContextApplication(
                requested=bool(context),
                applied=bool(context),
                method="mock_prompt" if context else None,
                input_chars=len(context),
                used_chars=len(context),
            ),
        )


def _audio_label(audio: AudioInput) -> str:
    if isinstance(audio, AudioPath):
        return hashlib.sha256(f"{audio.path}:{audio.start}:{audio.end}".encode()).hexdigest()[:12]
    if isinstance(audio, AudioComposition):
        return hashlib.sha256(repr(audio).encode()).hexdigest()[:12]
    return hashlib.sha256(audio).hexdigest()[:12]
