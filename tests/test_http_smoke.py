from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("ASR_BASE_URL") is None,
    reason="set ASR_BASE_URL to run HTTP smoke tests against a live server",
)


BASE_URL = os.getenv("ASR_BASE_URL", "http://127.0.0.1:18080").rstrip("/")
FIXTURE_AUDIO = Path("test-fixtures/audio/test_short.wav")


def error_code(response: httpx.Response) -> str | None:
    try:
        body: dict[str, Any] = response.json()
    except ValueError:
        return None
    error = body.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    return code if isinstance(code, str) else None


def test_live_health_and_models() -> None:
    with httpx.Client(timeout=10.0, trust_env=False) as client:
        health = client.get(f"{BASE_URL}/health")
        models = client.get(f"{BASE_URL}/v1/models")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert models.status_code == 200
    model_ids = {model["id"] for model in models.json()["models"]}
    assert {"qwen3-asr-0.6b", "qwen3-asr-1.7b"}.issubset(model_ids)


def test_live_transcription_and_timestamp_capability() -> None:
    with httpx.Client(timeout=httpx.Timeout(connect=5.0, read=1800.0, write=30.0, pool=5.0), trust_env=False) as client:
        models_response = client.get(f"{BASE_URL}/v1/models")
        models_response.raise_for_status()
        qwen = next(model for model in models_response.json()["models"] if model["id"] == "qwen3-asr-1.7b")
        timestamp_modes = qwen["capabilities"]["timestamps"]
        with FIXTURE_AUDIO.open("rb") as audio_file:
            transcription = client.post(
                f"{BASE_URL}/v1/audio/transcriptions",
                files={"file": (FIXTURE_AUDIO.name, audio_file, "audio/wav")},
                data={
                    "model": "qwen3-asr-1.7b",
                    "backend": "auto",
                    "language": "auto",
                    "response_format": "json",
                    "timestamps": "none",
                },
            )

        with FIXTURE_AUDIO.open("rb") as audio_file:
            timestamped = client.post(
                f"{BASE_URL}/v1/audio/transcriptions",
                files={"file": (FIXTURE_AUDIO.name, audio_file, "audio/wav")},
                data={
                    "model": "qwen3-asr-1.7b",
                    "backend": "auto",
                    "language": "auto",
                    "response_format": "json",
                    "timestamps": "word",
                },
            )

    assert transcription.status_code == 200
    assert transcription.json()["text"]
    if "word" in timestamp_modes:
        assert timestamped.status_code == 200
        body = timestamped.json()
        assert body["timestamps"]
        assert all(item["source"] == "qwen3_forced_aligner" for item in body["timestamps"])
    else:
        assert timestamped.status_code == 422
        assert error_code(timestamped) == "capability_not_supported"
