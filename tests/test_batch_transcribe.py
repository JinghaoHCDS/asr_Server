from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from scripts.batch_transcribe import BatchRunner


class FakeGateway:
    def __init__(self) -> None:
        self.submissions: list[tuple[Path, dict[str, str], str]] = []
        self.jobs: dict[str, dict[str, Any]] = {}

    def models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "qwen3-asr-1.7b",
                "default": True,
                "revision": "qwen-revision",
                "capabilities": {
                    "transcription": True,
                    "forced_alignment": True,
                    "timestamps": ["auto", "word", "char"],
                    "diarization": False,
                },
            },
            {
                "id": "moss-transcribe-diarize-0.9b",
                "default": False,
                "revision": "moss-revision",
                "capabilities": {
                    "transcription": True,
                    "forced_alignment": False,
                    "timestamps": [],
                    "diarization": True,
                    "speaker_resolution_modes": ["off", "auto", "required"],
                },
            },
        ]

    def submit(self, path: Path, fields: dict[str, str]) -> dict[str, Any]:
        job_id = f"job-{len(self.submissions):04d}"
        self.submissions.append((path, dict(fields), job_id))
        purpose = fields["purpose"]
        result: dict[str, Any] = {
            "model": fields["model"],
            "duration": 1.0,
            "text": f"text:{path.read_text(encoding='utf-8')}",
            "warnings": [],
            "context": {"applied": bool(fields.get("context") or fields.get("hotwords"))},
            "timing": {"reading_segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]},
            "segments": [{"speaker": "speaker-0001", "text": "hello"}],
            "purpose": purpose,
        }
        self.jobs[job_id] = result
        return {"id": job_id, "status": "queued"}

    def job(self, job_id: str, *, detail: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": job_id, "status": "completed", "progress": {"percent": 100.0}}
        if detail == "full":
            payload["result"] = self.jobs[job_id]
        return payload


class FailOnceGateway(FakeGateway):
    def job(self, job_id: str, *, detail: str) -> dict[str, Any]:
        if job_id == "job-0000":
            return {
                "id": job_id,
                "status": "failed",
                "progress": {"percent": 50.0},
                "error": {"code": "inference_failed", "message": "synthetic failure"},
            }
        return super().job(job_id, detail=detail)


class SubmitTimeoutOnceGateway(FakeGateway):
    def __init__(self) -> None:
        super().__init__()
        self.attempted_fields: list[dict[str, str]] = []

    def submit(self, path: Path, fields: dict[str, str]) -> dict[str, Any]:
        self.attempted_fields.append(dict(fields))
        if len(self.attempted_fields) == 1:
            raise httpx.ReadTimeout("synthetic submit timeout")
        return super().submit(path, fields)


def _manifest(tmp_path: Path, count: int) -> Path:
    items = []
    for index in range(count):
        media = tmp_path / f"audio-{index:03d}.wav"
        media.write_text(f"audio-content-{index}", encoding="utf-8")
        items.append(
            {
                "id": f"item-{index:03d}",
                "path": media.name,
                "title": f"Title {index}",
                "source": f"source-{index}",
                "context": f"context-{index}",
                "keywords": [f"TERM-{index}"],
                "purpose": "oral_analysis" if index % 2 == 0 else "podcast",
            }
        )
    path = tmp_path / "batch.json"
    path.write_text(json.dumps({"items": items}, ensure_ascii=False), encoding="utf-8")
    return path


def test_batch_of_300_is_rate_limited_paired_and_resumable(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, 300)
    output = tmp_path / "output"
    gateway = FakeGateway()

    state = BatchRunner(
        gateway,
        manifest,
        output,
        max_in_flight=7,
        poll_seconds=0,
    ).run()

    assert len(gateway.submissions) == 300
    assert all(item["status"] == "completed" for item in state["items"])
    for path, fields, _job_id in gateway.submissions:
        index = int(path.stem.split("-")[1])
        assert fields["context"] == f"context-{index}"
        assert json.loads(fields["hotwords"]) == [f"TERM-{index}"]
        expected_model = "qwen3-asr-1.7b" if index % 2 == 0 else "moss-transcribe-diarize-0.9b"
        assert fields["model"] == expected_model
    assert len(list((output / "items").glob("*/transcript.json"))) == 300
    assert (output / "index.md").is_file()

    resumed_gateway = FakeGateway()
    resumed = BatchRunner(
        resumed_gateway,
        manifest,
        output,
        max_in_flight=7,
        poll_seconds=0,
    ).run()
    assert resumed_gateway.submissions == []
    assert all(item["status"] == "completed" for item in resumed["items"])


def test_retry_failed_is_explicit_and_keeps_stable_item_id(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, 1)
    output = tmp_path / "output"
    gateway = FakeGateway()
    first = BatchRunner(gateway, manifest, output, poll_seconds=0).run()
    item_id = first["items"][0]["id"]
    saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    saved["items"][0]["status"] = "failed"
    saved["items"][0]["attempts"] = 2
    (output / "manifest.json").write_text(json.dumps(saved), encoding="utf-8")

    retry_gateway = FakeGateway()
    retried = BatchRunner(
        retry_gateway,
        manifest,
        output,
        poll_seconds=0,
        retry_failed=True,
    ).run()

    assert len(retry_gateway.submissions) == 1
    assert retried["items"][0]["id"] == item_id
    assert retried["items"][0]["status"] == "completed"


def test_compute_retry_uses_a_new_idempotency_key(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, 1)
    gateway = FailOnceGateway()

    state = BatchRunner(gateway, manifest, tmp_path / "output", poll_seconds=0).run()

    assert state["items"][0]["status"] == "completed"
    assert len(gateway.submissions) == 2
    first_key = gateway.submissions[0][1]["idempotency_key"]
    retry_key = gateway.submissions[1][1]["idempotency_key"]
    assert retry_key == f"{first_key}:compute-1"


def test_submit_transport_retry_reuses_the_same_idempotency_key(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, 1)
    gateway = SubmitTimeoutOnceGateway()

    state = BatchRunner(gateway, manifest, tmp_path / "output", poll_seconds=0).run()

    assert state["items"][0]["status"] == "completed"
    assert len(gateway.attempted_fields) == 2
    assert gateway.attempted_fields[0]["idempotency_key"] == gateway.attempted_fields[1]["idempotency_key"]
