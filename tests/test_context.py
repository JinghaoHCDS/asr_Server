from __future__ import annotations

import pytest

from asr_server.context import prepare_context
from asr_server.errors import AsrError


def test_context_normalizes_deduplicates_and_preserves_product_names() -> None:
    prepared = prepare_context(
        "  项目使用 Qwen3-ASR\x00 与 RTX 5070 Ti。  ",
        '["Qwen3-ASR", "qwen3-asr", "Hugging Face", "CUDA", "CUDA"]',
    )

    assert prepared.requested is True
    assert prepared.hotword_count == 3
    assert "Qwen3-ASR" in prepared.prompt
    assert "Hugging Face" in prepared.prompt
    assert "CUDA" in prepared.prompt
    assert "do not invent listed terms" in prepared.prompt
    assert "\x00" not in prepared.prompt


def test_context_truncation_is_visible_instead_of_silent() -> None:
    prepared = prepare_context("x" * 5_000, ",".join(f"term-{index}" for index in range(120)))

    assert prepared.truncated is True
    assert prepared.used_chars <= 4_000
    assert 0 < prepared.hotword_count <= 100
    assert "context_truncated:body" in prepared.warnings
    assert "context_truncated:hotword_count" in prepared.warnings
    assert "context_truncated:total" in prepared.warnings


def test_hotwords_reject_non_string_json_members() -> None:
    with pytest.raises(AsrError) as exc_info:
        prepare_context("", '["ok", 1]')

    assert exc_info.value.code == "bad_request"
