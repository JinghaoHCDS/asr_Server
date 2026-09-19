from __future__ import annotations

import pytest

from scripts.asr_client import choose_backend, choose_fallback_model, choose_model_for_purpose


MODELS = [
    {
        "id": "qwen3-asr-1.7b",
        "capabilities": {
            "transcription": True,
            "backends": ["transformers"],
        },
    },
    {
        "id": "qwen3-asr-0.6b",
        "capabilities": {
            "transcription": True,
            "backends": ["transformers"],
        },
    },
]


def test_choose_backend_uses_declared_backend_when_requested_backend_is_unsupported() -> None:
    assert choose_backend(MODELS, "qwen3-asr-1.7b", "vllm") == "transformers"


def test_choose_backend_preserves_auto() -> None:
    assert choose_backend(MODELS, "qwen3-asr-1.7b", "auto") == "auto"


def test_choose_fallback_model_prefers_other_qwen_model() -> None:
    assert choose_fallback_model(MODELS, "qwen3-asr-1.7b") == "qwen3-asr-0.6b"


def test_purpose_selection_requires_discovered_capabilities() -> None:
    models = [
        {
            "id": "aligned",
            "capabilities": {"forced_alignment": True, "timestamps": ["word"]},
        },
        {
            "id": "podcast",
            "capabilities": {"diarization": True, "speaker_resolution_modes": ["auto"]},
        },
    ]

    assert choose_model_for_purpose(models, "oral_analysis", None) == "aligned"
    assert choose_model_for_purpose(models, "podcast", None) == "podcast"
    with pytest.raises(RuntimeError):
        choose_model_for_purpose(models, "oral_analysis", "podcast")
