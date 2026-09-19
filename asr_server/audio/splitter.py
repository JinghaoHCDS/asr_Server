from __future__ import annotations

import io
import importlib
import math
import wave
import warnings
import threading
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from asr_server.adapters.base import AudioPath
from asr_server.audio.metadata import AudioMetadata, inspect_audio, inspect_audio_path
from asr_server.errors import AsrError


with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop


SplitStrategy = Literal["auto", "none", "fixed", "silero", "energy"]

DEFAULT_SOFT_CHUNK_SECONDS = 120.0
DEFAULT_HARD_CHUNK_SECONDS = 300.0
DEFAULT_OVERLAP_SECONDS = 2.0
MAX_AUDIO_SECONDS_PER_FILE = 21_600.0
MAX_CHUNKS_PER_FILE = 4096
VAD_FRAME_SECONDS = 0.03
VAD_MIN_SPEECH_SECONDS = 0.15
VAD_MIN_SILENCE_SECONDS = 0.3
VAD_RELATIVE_THRESHOLD = 0.10
VAD_ABSOLUTE_THRESHOLD = 300.0


class SplitCancelled(Exception):
    pass


@dataclass(frozen=True)
class AudioChunk:
    index: int
    start: float
    end: float
    audio: bytes

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


@dataclass(frozen=True)
class SplitResult:
    strategy: str
    requested_strategy: str
    chunks: list[AudioChunk]
    metadata: AudioMetadata
    soft_chunk_seconds: float
    hard_chunk_seconds: float
    overlap_seconds: float
    vad_backend: str | None = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "requested_strategy": self.requested_strategy,
            "vad_backend": self.vad_backend,
            "chunk_count": len(self.chunks),
            "soft_chunk_seconds": self.soft_chunk_seconds,
            "hard_chunk_seconds": self.hard_chunk_seconds,
            "overlap_seconds": self.overlap_seconds,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class ChunkDescriptor:
    index: int
    start: float
    end: float
    audio_path: Path

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)

    def as_audio_input(self) -> AudioPath:
        return AudioPath(path=self.audio_path, start=self.start, end=self.end)


@dataclass(frozen=True)
class PathSplitResult:
    strategy: str
    requested_strategy: str
    chunks: list[ChunkDescriptor]
    metadata: AudioMetadata
    soft_chunk_seconds: float
    hard_chunk_seconds: float
    overlap_seconds: float
    vad_backend: str | None = None
    speech_intervals: tuple[tuple[float, float], ...] = ()
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "requested_strategy": self.requested_strategy,
            "vad_backend": self.vad_backend,
            "speech_interval_count": len(self.speech_intervals),
            "chunk_count": len(self.chunks),
            "soft_chunk_seconds": self.soft_chunk_seconds,
            "hard_chunk_seconds": self.hard_chunk_seconds,
            "overlap_seconds": self.overlap_seconds,
            "warnings": self.warnings,
        }


def split_audio(
    audio: bytes,
    *,
    split_strategy: str,
    max_chunk_seconds: float | None,
    overlap_seconds: float | None,
    hard_chunk_seconds: float = DEFAULT_HARD_CHUNK_SECONDS,
) -> SplitResult:
    requested_strategy = split_strategy
    strategy = _parse_strategy(split_strategy)
    metadata = inspect_audio(audio)
    if metadata.duration_seconds > MAX_AUDIO_SECONDS_PER_FILE:
        raise AsrError(
            422,
            "duration_limit_exceeded",
            "audio is longer than the server file duration limit",
            {"duration_seconds": metadata.duration_seconds, "max_audio_seconds_per_file": MAX_AUDIO_SECONDS_PER_FILE},
        )

    chunk_seconds = max_chunk_seconds or DEFAULT_SOFT_CHUNK_SECONDS
    overlap = min(DEFAULT_OVERLAP_SECONDS, chunk_seconds / 10) if overlap_seconds is None else overlap_seconds
    _validate_chunk_options(chunk_seconds, overlap, hard_chunk_seconds)

    if strategy == "none":
        if metadata.duration_seconds > hard_chunk_seconds:
            raise AsrError(
                422,
                "duration_limit_exceeded",
                "split_strategy=none exceeds the model hard chunk limit",
                {"duration_seconds": metadata.duration_seconds, "hard_chunk_seconds": hard_chunk_seconds},
            )
        return _single_chunk(
            audio,
            metadata,
            requested_strategy=requested_strategy,
            overlap_seconds=overlap,
            hard_chunk_seconds=hard_chunk_seconds,
        )

    if strategy == "auto" and metadata.duration_seconds <= chunk_seconds:
        return _single_chunk(
            audio,
            metadata,
            requested_strategy=requested_strategy,
            overlap_seconds=overlap,
            hard_chunk_seconds=hard_chunk_seconds,
        )

    warnings: list[str] = []
    if strategy in {"auto", "silero"}:
        silero_chunks, silero_warning = _try_split_silero(audio, metadata, chunk_seconds, overlap)
        if silero_warning is not None:
            warnings.append(silero_warning)
        if silero_chunks:
            return SplitResult(
                strategy="silero",
                requested_strategy=requested_strategy,
                chunks=silero_chunks,
                metadata=metadata,
                soft_chunk_seconds=DEFAULT_SOFT_CHUNK_SECONDS,
                hard_chunk_seconds=hard_chunk_seconds,
                overlap_seconds=overlap,
                vad_backend="silero",
                warnings=warnings,
            )

    if strategy in {"auto", "silero", "energy"}:
        energy_chunks = _split_energy(audio, metadata, chunk_seconds, overlap)
        if energy_chunks:
            return SplitResult(
                strategy="energy",
                requested_strategy=requested_strategy,
                chunks=energy_chunks,
                metadata=metadata,
                soft_chunk_seconds=DEFAULT_SOFT_CHUNK_SECONDS,
                hard_chunk_seconds=hard_chunk_seconds,
                overlap_seconds=overlap,
                vad_backend="energy",
                warnings=warnings,
            )
        warnings.append("energy_vad_no_speech_fallback")
        if metadata.duration_seconds <= chunk_seconds:
            return _single_chunk(
                audio,
                metadata,
                requested_strategy=requested_strategy,
                resolved_strategy="energy" if strategy == "energy" else "none",
                overlap_seconds=overlap,
                vad_backend="energy" if strategy == "energy" else None,
                warnings=warnings,
                hard_chunk_seconds=hard_chunk_seconds,
            )

    chunks = _split_wav(audio, metadata, chunk_seconds, overlap)
    if chunks is None:
        chunks = _split_raw(audio, metadata, chunk_seconds, overlap)
    return SplitResult(
        strategy="fixed",
        requested_strategy=requested_strategy,
        chunks=chunks,
        metadata=metadata,
        soft_chunk_seconds=DEFAULT_SOFT_CHUNK_SECONDS,
        hard_chunk_seconds=hard_chunk_seconds,
        overlap_seconds=overlap,
        warnings=warnings,
    )


def _parse_strategy(value: str) -> SplitStrategy:
    if value == "vad":
        return "silero"
    if value in {"auto", "none", "fixed", "silero", "energy"}:
        return cast(SplitStrategy, value)
    raise AsrError(
        400,
        "bad_request",
        f"unsupported split_strategy: {value}",
        {"supported": ["auto", "none", "fixed", "silero", "energy", "vad"]},
    )


def split_audio_path(
    audio_path: Path,
    *,
    split_strategy: str,
    max_chunk_seconds: float | None,
    overlap_seconds: float | None,
    cancel_event: threading.Event | None = None,
    hard_chunk_seconds: float = DEFAULT_HARD_CHUNK_SECONDS,
) -> PathSplitResult:
    requested_strategy = split_strategy
    strategy = _parse_strategy(split_strategy)
    metadata = inspect_audio_path(audio_path)
    if metadata.duration_seconds > MAX_AUDIO_SECONDS_PER_FILE:
        raise AsrError(
            422,
            "duration_limit_exceeded",
            "audio is longer than the server file duration limit",
            {"duration_seconds": metadata.duration_seconds, "max_audio_seconds_per_file": MAX_AUDIO_SECONDS_PER_FILE},
        )
    chunk_seconds = max_chunk_seconds or DEFAULT_SOFT_CHUNK_SECONDS
    overlap = min(DEFAULT_OVERLAP_SECONDS, chunk_seconds / 10) if overlap_seconds is None else overlap_seconds
    _validate_chunk_options(chunk_seconds, overlap, hard_chunk_seconds)
    warnings: list[str] = []
    vad_backend: str | None = None
    speech_intervals: tuple[tuple[float, float], ...] = ()
    if strategy == "none":
        if metadata.duration_seconds > hard_chunk_seconds:
            raise AsrError(
                422,
                "duration_limit_exceeded",
                "split_strategy=none exceeds the model hard chunk limit",
                {"duration_seconds": metadata.duration_seconds, "hard_chunk_seconds": hard_chunk_seconds},
            )
        windows = [(0.0, metadata.duration_seconds)]
        resolved_strategy = "none"
    elif strategy == "auto" and metadata.duration_seconds <= chunk_seconds:
        windows = [(0.0, metadata.duration_seconds)]
        resolved_strategy = "none"
    else:
        intervals: list[tuple[float, float]] = []
        if strategy in {"auto", "silero", "energy"}:
            if strategy in {"auto", "silero"}:
                warnings.append("silero_streaming_not_validated_fallback_to_energy")
            intervals = _energy_intervals_path(
                audio_path,
                metadata,
                padding_seconds=overlap,
                cancel_event=cancel_event,
            )
        if intervals:
            speech_intervals = tuple(intervals)
            windows = _pack_vad_intervals(intervals, chunk_seconds, overlap)
            resolved_strategy = "energy"
            vad_backend = "energy"
        else:
            if strategy in {"auto", "silero", "energy"}:
                warnings.append("energy_vad_no_speech_fallback")
            windows = _chunk_windows(metadata.duration_seconds, chunk_seconds, overlap)
            resolved_strategy = "fixed"
    _validate_window_count(windows)
    return PathSplitResult(
        strategy=resolved_strategy,
        requested_strategy=requested_strategy,
        chunks=[ChunkDescriptor(index=i, start=start, end=end, audio_path=audio_path) for i, (start, end) in enumerate(windows)],
        metadata=metadata,
        soft_chunk_seconds=DEFAULT_SOFT_CHUNK_SECONDS,
        hard_chunk_seconds=hard_chunk_seconds,
        overlap_seconds=overlap,
        vad_backend=vad_backend,
        speech_intervals=speech_intervals,
        warnings=warnings,
    )


def _validate_chunk_options(max_chunk_seconds: float, overlap_seconds: float, hard_chunk_seconds: float) -> None:
    if max_chunk_seconds <= 0:
        raise AsrError(400, "bad_request", "max_chunk_seconds must be greater than 0")
    if max_chunk_seconds > hard_chunk_seconds:
        raise AsrError(
            422,
            "duration_limit_exceeded",
            "max_chunk_seconds exceeds the server hard chunk limit",
            {"max_chunk_seconds": max_chunk_seconds, "hard_chunk_seconds": hard_chunk_seconds},
        )
    if overlap_seconds < 0:
        raise AsrError(400, "bad_request", "overlap_seconds must be greater than or equal to 0")
    if overlap_seconds >= max_chunk_seconds:
        raise AsrError(
            400,
            "bad_request",
            "overlap_seconds must be smaller than max_chunk_seconds",
        )


def _single_chunk(
    audio: bytes,
    metadata: AudioMetadata,
    *,
    requested_strategy: str,
    overlap_seconds: float,
    hard_chunk_seconds: float = DEFAULT_HARD_CHUNK_SECONDS,
    resolved_strategy: str = "none",
    vad_backend: str | None = None,
    warnings: list[str] | None = None,
) -> SplitResult:
    return SplitResult(
        strategy=resolved_strategy,
        requested_strategy=requested_strategy,
        chunks=[AudioChunk(index=0, start=0.0, end=metadata.duration_seconds, audio=audio)],
        metadata=metadata,
        soft_chunk_seconds=DEFAULT_SOFT_CHUNK_SECONDS,
        hard_chunk_seconds=hard_chunk_seconds,
        overlap_seconds=overlap_seconds,
        vad_backend=vad_backend,
        warnings=warnings or [],
    )


def _chunk_windows(duration_seconds: float, chunk_seconds: float, overlap_seconds: float) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_seconds:
        end = min(start + chunk_seconds, duration_seconds)
        windows.append((start, end))
        if end >= duration_seconds:
            break
        start = end - overlap_seconds
        if len(windows) > MAX_CHUNKS_PER_FILE:
            _validate_window_count(windows)
    return windows


def _validate_window_count(windows: list[tuple[float, float]]) -> None:
    if len(windows) > MAX_CHUNKS_PER_FILE:
        raise AsrError(
            422,
            "duration_limit_exceeded",
            "audio split would create too many chunks",
            {"chunk_count": len(windows), "max_chunks_per_file": MAX_CHUNKS_PER_FILE},
        )


def _interval_windows(
    start: float,
    end: float,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    cursor = start
    while cursor < end:
        window_end = min(cursor + chunk_seconds, end)
        windows.append((cursor, window_end))
        if window_end >= end:
            break
        cursor = window_end - overlap_seconds
    return windows


def _try_split_silero(
    audio: bytes,
    metadata: AudioMetadata,
    chunk_seconds: float,
    overlap_seconds: float,
) -> tuple[list[AudioChunk], str | None]:
    try:
        intervals = _silero_intervals(audio, metadata, padding_seconds=overlap_seconds)
    except _SileroUnavailable as exc:
        return [], f"silero_vad_unavailable: {exc}"
    except Exception as exc:
        return [], f"silero_vad_failed: {type(exc).__name__}"
    if not intervals:
        return [], "silero_vad_no_speech_fallback"
    windows = _pack_vad_intervals(intervals, chunk_seconds, overlap_seconds)
    chunks = _slice_wav_windows(audio, metadata, windows) or []
    if not chunks:
        return [], "silero_vad_slice_failed_fallback"
    return chunks, None


def _split_energy(
    audio: bytes,
    metadata: AudioMetadata,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[AudioChunk]:
    intervals = _energy_intervals(audio, metadata, padding_seconds=overlap_seconds)
    if not intervals:
        return []
    windows = _pack_vad_intervals(intervals, chunk_seconds, overlap_seconds)
    return _slice_wav_windows(audio, metadata, windows) or []


class _SileroUnavailable(RuntimeError):
    pass


def _silero_intervals(
    audio: bytes,
    metadata: AudioMetadata,
    *,
    padding_seconds: float,
) -> list[tuple[float, float]]:
    if metadata.format != "wav":
        raise _SileroUnavailable("audio is not wav")
    if metadata.sample_rate not in {8_000, 16_000}:
        raise _SileroUnavailable("Silero VAD requires 8000 Hz or 16000 Hz wav input")
    try:
        with wave.open(io.BytesIO(audio), "rb") as source:
            if source.getsampwidth() != 2 or source.getnchannels() != 1:
                raise _SileroUnavailable("Silero VAD requires mono 16-bit PCM wav input")
            frame_rate = source.getframerate()
            pcm = source.readframes(source.getnframes())
    except (EOFError, wave.Error) as exc:
        raise _SileroUnavailable("wav decode failed") from exc

    samples = array("h")
    samples.frombytes(pcm)
    if not samples:
        return []

    try:
        torch = importlib.import_module("torch")
        silero_vad = importlib.import_module("silero_vad")
    except ModuleNotFoundError as exc:
        raise _SileroUnavailable("torch or silero_vad is not installed") from exc

    waveform = torch.tensor([sample / 32768.0 for sample in samples], dtype=torch.float32)
    model = _load_silero_model(silero_vad)
    get_speech_timestamps = getattr(silero_vad, "get_speech_timestamps", None)
    if not callable(get_speech_timestamps):
        raise _SileroUnavailable("silero_vad.get_speech_timestamps is not available")
    speech_timestamps = get_speech_timestamps(waveform, model, sampling_rate=frame_rate)
    intervals = []
    for item in speech_timestamps:
        if not isinstance(item, dict):
            continue
        start_sample = int(item.get("start", 0))
        end_sample = int(item.get("end", 0))
        if end_sample <= start_sample:
            continue
        start = max((start_sample / frame_rate) - padding_seconds, 0.0)
        end = min((end_sample / frame_rate) + padding_seconds, metadata.duration_seconds)
        intervals.append((start, end))
    return intervals


def _load_silero_model(silero_vad: Any) -> Any:
    load_silero_vad = getattr(silero_vad, "load_silero_vad", None)
    if not callable(load_silero_vad):
        raise _SileroUnavailable("silero_vad.load_silero_vad is not available")
    return load_silero_vad()


def _energy_intervals(
    audio: bytes,
    metadata: AudioMetadata,
    *,
    padding_seconds: float,
) -> list[tuple[float, float]]:
    if metadata.format != "wav":
        return []
    try:
        with wave.open(io.BytesIO(audio), "rb") as source:
            if source.getsampwidth() != 2 or source.getnchannels() != 1:
                return []
            frame_rate = source.getframerate()
            pcm = source.readframes(source.getnframes())
    except (EOFError, wave.Error):
        return []

    samples = array("h")
    samples.frombytes(pcm)
    if not samples:
        return []

    frame_samples = max(int(frame_rate * VAD_FRAME_SECONDS), 1)
    rms_values = []
    for offset in range(0, len(samples), frame_samples):
        frame = samples[offset : offset + frame_samples]
        if not frame:
            continue
        rms_values.append(math.sqrt(sum(sample * sample for sample in frame) / len(frame)))
    if not rms_values:
        return []

    threshold = max(VAD_ABSOLUTE_THRESHOLD, max(rms_values) * VAD_RELATIVE_THRESHOLD)
    speech_frames = [rms >= threshold for rms in rms_values]
    raw_intervals: list[tuple[float, float]] = []
    start_frame: int | None = None
    for index, is_speech in enumerate(speech_frames):
        if is_speech and start_frame is None:
            start_frame = index
        if not is_speech and start_frame is not None:
            raw_intervals.append((start_frame * VAD_FRAME_SECONDS, index * VAD_FRAME_SECONDS))
            start_frame = None
    if start_frame is not None:
        raw_intervals.append((start_frame * VAD_FRAME_SECONDS, len(speech_frames) * VAD_FRAME_SECONDS))

    merged = _merge_vad_intervals(raw_intervals)
    padded = []
    for start, end in merged:
        if end - start < VAD_MIN_SPEECH_SECONDS:
            continue
        padded.append((max(start - padding_seconds, 0.0), min(end + padding_seconds, metadata.duration_seconds)))
    return padded


def _energy_intervals_path(
    audio_path: Path,
    metadata: AudioMetadata,
    *,
    padding_seconds: float,
    cancel_event: threading.Event | None = None,
) -> list[tuple[float, float]]:
    if metadata.format != "wav":
        return []
    try:
        with wave.open(str(audio_path), "rb") as source:
            if source.getsampwidth() != 2 or source.getnchannels() != 1:
                return []
            frame_rate = source.getframerate()
            frame_samples = max(int(frame_rate * VAD_FRAME_SECONDS), 1)
            rms_values: list[float] = []
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise SplitCancelled()
                pcm = source.readframes(frame_samples)
                if not pcm:
                    break
                rms_values.append(float(audioop.rms(pcm, 2)))
    except (EOFError, wave.Error):
        return []
    if not rms_values:
        return []
    threshold = max(VAD_ABSOLUTE_THRESHOLD, max(rms_values) * VAD_RELATIVE_THRESHOLD)
    raw_intervals: list[tuple[float, float]] = []
    start_frame: int | None = None
    for index, rms in enumerate(rms_values):
        is_speech = rms >= threshold
        if is_speech and start_frame is None:
            start_frame = index
        elif not is_speech and start_frame is not None:
            raw_intervals.append((start_frame * VAD_FRAME_SECONDS, index * VAD_FRAME_SECONDS))
            start_frame = None
    if start_frame is not None:
        raw_intervals.append((start_frame * VAD_FRAME_SECONDS, len(rms_values) * VAD_FRAME_SECONDS))
    padded = []
    for start, end in _merge_vad_intervals(raw_intervals):
        if end - start >= VAD_MIN_SPEECH_SECONDS:
            padded.append((max(start - padding_seconds, 0.0), min(end + padding_seconds, metadata.duration_seconds)))
    return padded


def _merge_vad_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if not merged or start - merged[-1][1] > VAD_MIN_SILENCE_SECONDS:
            merged.append((start, end))
            continue
        previous_start, _previous_end = merged[-1]
        merged[-1] = (previous_start, end)
    return merged


def _pack_vad_intervals(
    intervals: list[tuple[float, float]],
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    current_start: float | None = None
    current_end: float | None = None
    for interval_start, interval_end in intervals:
        if interval_end - interval_start > chunk_seconds:
            if current_start is not None and current_end is not None:
                windows.append((current_start, current_end))
                current_start = None
                current_end = None
            windows.extend(_interval_windows(interval_start, interval_end, chunk_seconds, overlap_seconds))
            continue
        if current_start is None or current_end is None:
            current_start = interval_start
            current_end = interval_end
            continue
        if interval_end - current_start <= chunk_seconds:
            current_end = interval_end
            continue
        windows.append((current_start, current_end))
        current_start = max(interval_start - overlap_seconds, 0.0)
        current_end = interval_end
    if current_start is not None and current_end is not None:
        windows.append((current_start, current_end))
    _validate_window_count(windows)
    return windows


def _split_raw(
    audio: bytes,
    metadata: AudioMetadata,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[AudioChunk]:
    bytes_per_second = len(audio) / metadata.duration_seconds
    chunks = []
    for index, (start, end) in enumerate(_chunk_windows(metadata.duration_seconds, chunk_seconds, overlap_seconds)):
        start_byte = int(start * bytes_per_second)
        end_byte = max(start_byte + 1, int(end * bytes_per_second))
        chunks.append(AudioChunk(index=index, start=start, end=end, audio=audio[start_byte:end_byte]))
    return chunks


def _split_wav(
    audio: bytes,
    metadata: AudioMetadata,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[AudioChunk] | None:
    if metadata.format != "wav":
        return None
    return _slice_wav_windows(audio, metadata, _chunk_windows(metadata.duration_seconds, chunk_seconds, overlap_seconds))


def _slice_wav_windows(
    audio: bytes,
    metadata: AudioMetadata,
    windows: list[tuple[float, float]],
) -> list[AudioChunk] | None:
    if metadata.format != "wav":
        return None
    try:
        with wave.open(io.BytesIO(audio), "rb") as source:
            frame_rate = source.getframerate()
            params = source.getparams()
            chunks = []
            for index, (start, end) in enumerate(windows):
                start_frame = int(start * frame_rate)
                end_frame = min(int(end * frame_rate), source.getnframes())
                source.setpos(start_frame)
                frames = source.readframes(max(end_frame - start_frame, 0))
                output = io.BytesIO()
                with wave.open(output, "wb") as target:
                    target.setparams(params)
                    target.writeframes(frames)
                chunks.append(AudioChunk(index=index, start=start, end=end, audio=output.getvalue()))
    except (EOFError, wave.Error):
        return None
    return chunks
