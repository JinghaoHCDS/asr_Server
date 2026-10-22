"""Optional CPU evidence for ASR-empty intervals; never a proof of no speech.

The package and model are verified without importing silero_vad, whose loader
changes global torch threading. Missing or changed dependencies fail closed.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import math
import sys
import threading
import wave
from array import array
from pathlib import Path
from time import perf_counter, process_time
from typing import Any


_VERSION = "6.2.3"
_MODEL_SHA256 = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"
_SAMPLE_RATE = 16000
_FRAME_SAMPLES = 512
_THRESHOLD = 0.5
_TARGET_PEAK = 0.9
_MAX_GAIN = 100
_TIMEOUT_SECONDS = 60.0
_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: tuple[tuple[str, str], Any, Any] | None = None


def _identity(*, available: bool) -> dict[str, object]:
    return {
        "available": available, "detector": "silero-vad", "version": _VERSION,
        "model_sha256": _MODEL_SHA256, "sample_rate": _SAMPLE_RATE,
        "frame_samples": _FRAME_SAMPLES, "threshold": _THRESHOLD,
        "target_peak": _TARGET_PEAK, "max_gain": _MAX_GAIN,
        "timeout_seconds": _TIMEOUT_SECONDS,
    }


def _verified_model() -> tuple[dict[str, object], Path] | None:
    try:
        distribution = importlib.metadata.distribution("silero-vad")
        if distribution.version != _VERSION:
            return None
        path = Path(str(distribution.locate_file("silero_vad/data/silero_vad.jit")))
        digest = hashlib.sha256()
        with path.open("rb") as model_file:
            while block := model_file.read(1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != _MODEL_SHA256:
            return None
        return _identity(available=True), path
    except Exception:
        return None


def detector_identity() -> dict[str, object]:
    """Describe the exact installed detector without importing model runtimes."""
    verified = _verified_model()
    return verified[0] if verified is not None else _identity(available=False)


def _read_interval(path: Path, start: float, end: float) -> array[int] | None:
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or not 0 < end - start <= 180:
        return None
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or audio.getframerate() != _SAMPLE_RATE:
            return None
        frames = audio.getnframes()
        if end > frames / _SAMPLE_RATE + 1e-9:
            return None
        first, stop = round(start * _SAMPLE_RATE), round(end * _SAMPLE_RATE)
        count = stop - first
        if not 0 < count <= 180 * _SAMPLE_RATE or stop > frames:
            return None
        audio.setpos(first)
        raw = audio.readframes(count)
        if len(raw) != count * 2:
            return None
    samples = array("h", raw)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def non_speech_evidence(
    path: Path, start: float, end: float,
    expected_identity: dict[str, object] | None = None,
) -> dict[str, object] | None:
    """Return reviewable negative VAD evidence only after complete dual scans.

    Call only after ASR and its bounded acoustic reviews still returned no text.
    The caller must preserve the source interval and mark it needs_review.
    """
    global _MODEL_CACHE
    started = process_time()
    wall_started = perf_counter()
    deadline = wall_started + _TIMEOUT_SECONDS
    try:
        verified = _verified_model()
        if verified is None:
            return None
        identity, model_path = verified
        if expected_identity is not None and expected_identity != identity:
            return None
        samples = _read_interval(path, start, end)
        if samples is None:
            return None
        peak = max(abs(value) for value in samples)
        if not peak:  # Exact zero and near-zero tails have separate policies.
            return None
        gain = max(1.0, _TARGET_PEAK / (peak / 32768.0))
        if not math.isfinite(gain) or gain > _MAX_GAIN:
            return None
        frames_per_view = math.ceil(len(samples) / _FRAME_SAMPLES)
        remaining = deadline - perf_counter()
        if remaining <= 0 or not _MODEL_LOCK.acquire(timeout=remaining):
            return None
        try:
            # Check again after waiting for a shared, stateful CPU model.
            current = _verified_model()
            if current is None or current[0] != identity or current[1] != model_path:
                return None
            key = (str(model_path), _MODEL_SHA256)
            if _MODEL_CACHE is None or _MODEL_CACHE[0] != key:
                torch = importlib.import_module("torch")
                model = torch.jit.load(str(model_path), map_location="cpu").eval()
                _MODEL_CACHE = (key, model, torch)
            _key, model, torch = _MODEL_CACHE
            with torch.no_grad():
                waveform = torch.frombuffer(samples, dtype=torch.int16).to(dtype=torch.float32).div_(32768.0)
                for multiplier in (1.0, gain):
                    model.reset_states()
                    processed = 0
                    for offset in range(0, len(samples), _FRAME_SAMPLES):
                        if perf_counter() >= deadline:
                            return None
                        frame = waveform[offset:offset + _FRAME_SAMPLES] * multiplier
                        if frame.numel() < _FRAME_SAMPLES:
                            frame = torch.nn.functional.pad(frame, (0, _FRAME_SAMPLES - frame.numel()))
                        probability = float(model(frame, _SAMPLE_RATE).item())
                        if perf_counter() >= deadline:
                            return None
                        if not math.isfinite(probability) or not 0 <= probability <= 1 or probability >= _THRESHOLD:
                            return None
                        processed += 1
                    if processed != frames_per_view:
                        return None
            # A dependency swap during inference cannot authorize delivery.
            current = _verified_model()
            if current is None or current[0] != identity or current[1] != model_path:
                return None
            if perf_counter() >= deadline:
                return None
        finally:
            _MODEL_LOCK.release()
        return {
            "method": "silero_no_speech_detected", "detector_identity": identity,
            "samples": len(samples), "frames_per_view": frames_per_view, "views": 2,
            "peak_pcm16": peak, "gain": gain,
            "cpu_ms": max(0.0, (process_time() - started) * 1000),
            "elapsed_ms": max(0.0, (perf_counter() - wall_started) * 1000),
        }
    except Exception:
        return None
