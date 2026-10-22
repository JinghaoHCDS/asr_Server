from __future__ import annotations

import pytest
from asr_server.audio.subtitles import build_reading_segments
from asr_server.errors import AsrError


def test_readable_cues_preserve_words_punctuation_and_real_times() -> None:
    text = "你好，世界！你好，世界！"
    words = ["你", "好", "世", "界", "你", "好", "世", "界"]
    stamps = [{"text": c, "start": i * .5, "end": i * .5 + .4, "source": "qwen3_forced_aligner"}
              for i, c in enumerate(words)]
    groups = build_reading_segments(text, stamps)
    assert [s["text"] for s in groups] == ["你好，世界！", "你好，世界！"]
    assert groups[1]["start"] == 2
    assert groups[1]["end"] == 3.9
    assert [i for s in groups for i in s["timestamps"]] == stamps


def test_english_spaces_and_punctuation_are_not_guessed() -> None:
    words = ["Hello", "world", "use", "Qwen3-ASR"]
    stamps = [{"text": c, "start": i, "end": i + .5} for i, c in enumerate(words)]
    groups = build_reading_segments("Hello world, use Qwen3-ASR!", stamps, max_chars=64)
    assert groups[0]["text"] == "Hello world, use Qwen3-ASR!"


def test_missing_alignment_never_silently_deletes_text() -> None:
    with pytest.raises(AsrError, match="aligned units"):
        build_reading_segments("你好世界", [{"text": "你好", "start": 0, "end": 1}])
    with pytest.raises(AsrError, match="ordered"):
        build_reading_segments("你好", [{"text": "你", "start": 2, "end": 3}, {"text": "好", "start": 0, "end": 1}])
