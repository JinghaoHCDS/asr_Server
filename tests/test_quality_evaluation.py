from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.evaluate_quality import error_counts, normalize, percentile, read_manifest, score_run, score_sample


def test_cer_counts_do_not_hide_deletions_and_insertions() -> None:
    missing = error_counts("你好世界", "你好")
    repeated = error_counts("你好", "你好你好")
    assert missing["deletions"] == 2
    assert missing["cer"] == .5
    assert repeated["insertions"] == 2
    assert repeated["cer"] == 1
    assert error_counts("", "编造")["cer"] is None
    assert normalize("ＡＳＲ， 你好。") == "asr你好"


def test_term_recall_is_not_a_hit_count_and_alignment_cannot_reuse_units() -> None:
    case: dict[str, Any] = {"id": "a", "duration_seconds": 10, "reference_text": "Qwen Qwen",
        "terms": ["Qwen"], "absent_terms": ["ABSENT"],
        "alignment_reference": [{"index": 1, "text": "你好", "start": 0, "end": 1}]}
    score = score_sample(case, {"text": "Qwen ABSENT", "timestamps": [{"text": "你好", "start": 0, "end": 1}]})
    assert score["expected_terms"] == 2
    assert score["recovered_terms"] == 1
    assert score["absent_term_insertions"] == 1
    assert score["alignment_matched_units"] == 0
    assert percentile([.1, .2, .7], .95) == .7


def test_empty_corpus_is_not_quality_evidence(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps({"schema_version": 1, "items": []}))
    with pytest.raises(ValueError, match="no annotated"):
        read_manifest(manifest, "acceptance")


def test_mock_and_missing_duration_buckets_never_pass_release_gate(tmp_path: Path) -> None:
    (tmp_path / "run.json").write_text(json.dumps({"mock": True}))
    case: dict[str, Any] = {"id": "a", "duration_seconds": 13, "reference_text": "你好"}
    for profile in ["legacy", "quality_60_0"]:
        (tmp_path / profile).mkdir()
        (tmp_path / profile / "a.json").write_text(json.dumps({"text": "你好"}))
    report = score_run([case], tmp_path, ["legacy", "quality_60_0"], "acceptance")
    assert report["status"] == "not_accepted"
    assert report["selected_candidate"] is None


def test_public_corpus_import_freezes_source_reference_and_audio_hash(tmp_path: Path) -> None:
    import hashlib
    import subprocess
    import sys
    import wave
    root = tmp_path / "aishell"
    (root / "transcript").mkdir(parents=True)
    (root / "wav" / "dev" / "S0001").mkdir(parents=True)
    (root / "transcript" / "aishell_transcript_v0.8.txt").write_text("BAC001S0001W0001 你 好 世 界\n")
    audio = root / "wav" / "dev" / "S0001" / "BAC001S0001W0001.wav"
    with wave.open(str(audio), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"\x00\x00" * 16000)
    output = tmp_path / "imported"
    result = subprocess.run([sys.executable, "-m", "scripts.import_aishell", str(root), "--output", str(output)],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    cases = read_manifest(output / "corpus.json", "dev")
    assert cases[0]["reference_text"].strip() == "你好世界"
    assert cases[0]["sha256"] == hashlib.sha256(audio.read_bytes()).hexdigest()
    assert cases[0]["source"] == "https://www.openslr.org/33/"
