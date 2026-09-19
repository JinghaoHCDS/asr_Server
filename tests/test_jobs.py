from __future__ import annotations

import asyncio
import io
import wave
from array import array
from collections.abc import AsyncIterator
from pathlib import Path
from datetime import timedelta
from typing import Any, cast

import pytest
from httpx import ASGITransport, AsyncClient

from asr_server import main as main_module
from asr_server.adapters.base import TranscriptionResult
from asr_server.adapters.mock import MockAsrAdapter
from asr_server.config import Settings
from asr_server.errors import AsrError
from asr_server.main import create_app
from asr_server.jobs import utc_now


def make_wav(duration_seconds: float, amplitude: int = 10_000, sample_rate: int = 16_000) -> bytes:
    samples = array("h", [amplitude] * int(duration_seconds * sample_rate))
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    return output.getvalue()


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        yield client


async def create_job(
    client: AsyncClient,
    *,
    audio: bytes | None = None,
    data: dict[str, str] | None = None,
) -> dict[str, object]:
    response = await client.post(
        "/v1/audio/transcription-jobs",
        files={"file": ("sample.wav", audio or make_wav(0.1), "audio/wav")},
        data=data or {},
    )
    assert response.status_code == 202
    return cast(dict[str, object], response.json())


async def wait_for_status(
    client: AsyncClient,
    job_id: str,
    statuses: set[str],
    *,
    detail: str = "full",
) -> dict[str, object]:
    for _ in range(80):
        response = await client.get(f"/v1/jobs/{job_id}", params={"detail": detail})
        assert response.status_code == 200
        body = response.json()
        body = cast(dict[str, object], body)
        if body["status"] in statuses:
            return body
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach {statuses}")


def patch_job_tempdir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    counter = 0

    def mkdtemp(prefix: str) -> str:
        nonlocal counter
        counter += 1
        path = tmp_path / f"{prefix}{counter}"
        path.mkdir()
        return str(path)

    monkeypatch.setattr("asr_server.audio.workspace.tempfile.mkdtemp", mkdtemp)


async def test_create_job_returns_accepted_and_status_url(client: AsyncClient) -> None:
    body = await create_job(client)

    assert body["status"] == "queued"
    assert body["model"] == "qwen3-asr-1.7b"
    assert body["backend"] == "transformers"
    assert body["queue_position"] == 1
    assert body["status_url"] == f"/v1/jobs/{body['id']}"

    completed = await wait_for_status(client, str(body["id"]), {"completed"})
    completed_result = cast(dict[str, Any], completed["result"])
    completed_progress = cast(dict[str, Any], completed["progress"])
    assert completed_result["text"]
    assert completed_result["model"] == "qwen3-asr-1.7b"
    assert completed_progress["percent"] == 100.0


async def test_job_status_is_compact_by_default_and_full_is_explicit(client: AsyncClient) -> None:
    created = await create_job(client)
    await wait_for_status(client, str(created["id"]), {"completed"})

    compact = await client.get(f"/v1/jobs/{created['id']}")
    full = await client.get(f"/v1/jobs/{created['id']}", params={"detail": "full"})

    assert compact.status_code == 200
    assert "result" not in compact.json()
    assert "result" in full.json()


async def test_job_idempotency_deduplicates_submit_timeout_retry(client: AsyncClient) -> None:
    fields = {"idempotency_key": "same-material-and-parameters"}
    first = await create_job(client, data=fields)
    second = await create_job(client, data=fields)

    assert second["id"] == first["id"]
    assert second["deduplicated"] is True


async def test_job_idempotency_rejects_different_audio(client: AsyncClient) -> None:
    fields = {"idempotency_key": "conflicting-key"}
    await create_job(client, audio=make_wav(0.1), data=fields)
    response = await client.post(
        "/v1/audio/transcription-jobs",
        files={"file": ("different.wav", make_wav(0.2), "audio/wav")},
        data=fields,
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "idempotency_conflict"


async def test_purpose_profiles_select_runtime_capabilities_and_defaults() -> None:
    app = create_app(settings=Settings(enable_moss=True, enable_qwen_alignment=True))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        oral = await create_job(client, data={"purpose": "oral_analysis"})
        podcast = await create_job(client, data={"purpose": "podcast"})
        oral_full = await wait_for_status(client, str(oral["id"]), {"completed"})
        podcast_full = await wait_for_status(client, str(podcast["id"]), {"completed"})

    assert oral["model"] == "qwen3-asr-1.7b"
    oral_request = cast(dict[str, Any], oral_full["request"])
    podcast_request = cast(dict[str, Any], podcast_full["request"])
    assert oral_request["timestamps"] == "auto"
    assert oral_request["purpose"] == "oral_analysis"
    assert podcast["model"] == "moss-transcribe-diarize-0.9b"
    assert podcast_request["timestamps"] == "none"
    assert podcast_request["purpose"] == "podcast"
    result = cast(dict[str, Any], podcast_full["result"])
    assert result["execution"]["speaker_scope"] == "global"


async def test_completed_job_removes_uploaded_temp_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_job_tempdir(monkeypatch, tmp_path)
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        created = await create_job(client)
        await wait_for_status(client, str(created["id"]), {"completed"})

    assert list(tmp_path.iterdir()) == []


async def test_job_manager_start_removes_stale_job_temp_dirs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stale = tmp_path / "asr_job_stale_abc"
    stale.mkdir()
    (stale / "upload").write_bytes(b"old audio")
    unrelated = tmp_path / "other_temp"
    unrelated.mkdir()
    monkeypatch.setattr("asr_server.jobs.tempfile.gettempdir", lambda: str(tmp_path))

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver"):
        await app.state.job_manager.start()
        await app.state.job_manager.shutdown()

    assert not stale.exists()
    assert unrelated.exists()


async def test_multiple_jobs_are_serialized_and_show_queue_position() -> None:
    app = create_app(adapter_delay_seconds=0.08)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        first = await create_job(client)
        await wait_for_status(client, str(first["id"]), {"preprocessing", "splitting", "loading_model", "transcribing"})
        second = await create_job(client)

        queued = await client.get(f"/v1/jobs/{second['id']}")
        assert queued.status_code == 200
        assert queued.json()["status"] == "queued"
        assert queued.json()["queue_position"] == 1

        await wait_for_status(client, str(first["id"]), {"completed"})
        await wait_for_status(client, str(second["id"]), {"completed"})


async def test_job_reports_chunk_progress() -> None:
    app = create_app(adapter_delay_seconds=0.03)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        created = await create_job(
            client,
            audio=make_wav(0.11),
            data={
                "split_strategy": "fixed",
                "max_chunk_seconds": "0.04",
                "overlap_seconds": "0.01",
                "preserve_segments": "true",
            },
        )

        transcribing = await wait_for_status(client, str(created["id"]), {"transcribing"})
        progress = cast(dict[str, Any], transcribing["progress"])
        assert progress["total_chunks"] == 4
        assert progress["current_chunk"] >= 1
        assert progress["completed_chunks"] >= 0

        completed = await wait_for_status(client, str(created["id"]), {"completed"})
        completed_progress = cast(dict[str, Any], completed["progress"])
        completed_result = cast(dict[str, Any], completed["result"])
        assert completed_progress["total_chunks"] == 4
        assert completed_progress["completed_chunks"] == 4
        assert len(cast(list[dict[str, object]], completed_result["chunks"])) == 4


@pytest.mark.parametrize(
    ("model", "extra_data"),
    [
        ("qwen3-asr-1.7b", {}),
        ("moss-transcribe-diarize-0.9b", {"response_format": "verbose_json"}),
    ],
)
async def test_chunked_job_reports_cumulative_and_latest_chunk_text(
    model: str,
    extra_data: dict[str, str],
) -> None:
    app = create_app(settings=Settings(enable_moss=True), adapter_delay_seconds=0.03)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        created = await create_job(
            client,
            audio=make_wav(0.11),
            data={
                "model": model,
                "split_strategy": "fixed",
                "max_chunk_seconds": "0.04",
                "overlap_seconds": "0.01",
                "preserve_segments": "true",
                **extra_data,
            },
        )

        progress: dict[str, Any] | None = None
        for _ in range(80):
            response = await client.get(f"/v1/jobs/{created['id']}", params={"detail": "full"})
            assert response.status_code == 200
            body = cast(dict[str, Any], response.json())
            candidate = cast(dict[str, Any], body["progress"])
            if candidate.get("completed_chunks", 0) >= 1 and candidate.get("chunk_text"):
                progress = candidate
                break
            await asyncio.sleep(0.01)

        assert progress is not None
        assert progress["text"]
        assert progress["chunk_text"]
        assert str(progress["text"]).endswith(str(progress["chunk_text"]))
        chunks = cast(list[dict[str, Any]], progress["chunks"])
        assert chunks[-1] == {
            "index": progress["completed_chunks"],
            "text": progress["chunk_text"],
        }

        await wait_for_status(client, str(created["id"]), {"completed"})


async def test_moss_native_long_form_job_reports_stage_without_fake_chunk_progress() -> None:
    app = create_app(settings=Settings(enable_moss=True), adapter_delay_seconds=0.08)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        created = await create_job(
            client,
            data={
                "model": "moss-transcribe-diarize-0.9b",
                "response_format": "verbose_json",
            },
        )
        transcribing = await wait_for_status(client, str(created["id"]), {"transcribing"})
        completed = await wait_for_status(client, str(created["id"]), {"completed"})

    progress = cast(dict[str, Any], transcribing["progress"])
    split = cast(dict[str, Any], transcribing["split"])
    assert progress["phase"] == "transcribing"
    assert "total_chunks" not in progress
    assert "completed_chunks" not in progress
    assert "current_chunk" not in progress
    assert "text" not in progress
    assert "chunk_text" not in progress
    assert "chunks" not in progress
    assert split["execution_mode"] == "native_long_form"
    assert cast(dict[str, Any], completed["result"])["execution"]["speaker_scope"] == "global"


async def test_moss_native_long_form_cancellation_waits_for_model_invocation() -> None:
    app = create_app(settings=Settings(enable_moss=True), adapter_delay_seconds=0.08)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        created = await create_job(client, data={"model": "moss-transcribe-diarize-0.9b"})
        await wait_for_status(client, str(created["id"]), {"transcribing"})

        response = await client.delete(f"/v1/jobs/{created['id']}")
        cancelled = await wait_for_status(client, str(created["id"]), {"cancelled"})

    assert response.status_code == 200
    assert response.json()["status"] == "cancel_requested"
    assert response.json()["message"] == "cancellation will take effect after the current model invocation finishes"
    assert cancelled["status"] == "cancelled"


async def test_job_failure_uses_error_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_transcribe(
        self: MockAsrAdapter,
        audio: bytes,
        *,
        model_id: str,
        backend: str,
        language: str,
        context: str,
        max_new_tokens: int | None,
    ) -> TranscriptionResult:
        del self, audio, model_id, backend, language, context, max_new_tokens
        raise AsrError(503, "gpu_unavailable", "CUDA out of memory during Qwen ASR", {"phase": "transcribing"})

    monkeypatch.setattr(MockAsrAdapter, "transcribe", fail_transcribe)

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        created = await create_job(client)
        failed = await wait_for_status(client, str(created["id"]), {"failed"})

    assert failed["result"] is None
    error = cast(dict[str, Any], failed["error"])
    details = cast(dict[str, Any], error["details"])
    assert error["code"] == "gpu_unavailable"
    assert details["phase"] == "transcribing"


async def test_failed_job_removes_uploaded_temp_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_job_tempdir(monkeypatch, tmp_path)

    async def fail_transcribe(
        self: MockAsrAdapter,
        audio: bytes,
        *,
        model_id: str,
        backend: str,
        language: str,
        context: str,
        max_new_tokens: int | None,
    ) -> TranscriptionResult:
        del self, audio, model_id, backend, language, context, max_new_tokens
        raise AsrError(503, "gpu_unavailable", "CUDA out of memory during Qwen ASR", {"phase": "transcribing"})

    monkeypatch.setattr(MockAsrAdapter, "transcribe", fail_transcribe)

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        created = await create_job(client)
        await wait_for_status(client, str(created["id"]), {"failed"})

    assert list(tmp_path.iterdir()) == []


async def test_queued_job_can_be_cancelled() -> None:
    app = create_app(adapter_delay_seconds=0.08)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        first = await create_job(client)
        second = await create_job(client)

        response = await client.delete(f"/v1/jobs/{second['id']}")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

        cancelled = await client.get(f"/v1/jobs/{second['id']}")
        assert cancelled.json()["status"] == "cancelled"
        await wait_for_status(client, str(first["id"]), {"completed"})


async def test_cancelled_queued_job_removes_uploaded_temp_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_job_tempdir(monkeypatch, tmp_path)
    app = create_app(adapter_delay_seconds=0.08)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        first = await create_job(client)
        second = await create_job(client)

        response = await client.delete(f"/v1/jobs/{second['id']}")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

        await wait_for_status(client, str(first["id"]), {"completed"})

    assert list(tmp_path.iterdir()) == []


async def test_running_job_cancel_waits_for_chunk_boundary() -> None:
    app = create_app(adapter_delay_seconds=0.04)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        created = await create_job(
            client,
            audio=make_wav(0.11),
            data={"split_strategy": "fixed", "max_chunk_seconds": "0.04", "overlap_seconds": "0.01"},
        )
        await wait_for_status(client, str(created["id"]), {"transcribing"})

        response = await client.delete(f"/v1/jobs/{created['id']}")
        assert response.status_code == 200
        assert response.json()["status"] == "cancel_requested"

        cancelled = await wait_for_status(client, str(created["id"]), {"cancelled"})
        assert cancelled["status"] == "cancelled"
        progress = cast(dict[str, Any], cancelled["progress"])
        assert progress["phase"] == "cancelled"


async def test_unknown_job_returns_error_envelope(client: AsyncClient) -> None:
    response = await client.get("/v1/jobs/job_missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "job_not_found"


async def test_job_queue_limit_rejects_excess_jobs() -> None:
    app = create_app(settings=Settings(max_queued_jobs=1), adapter_delay_seconds=0.08)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        first = await create_job(client)

        response = await client.post(
            "/v1/audio/transcription-jobs",
            files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
        )

        assert response.status_code == 429
        assert response.json()["error"]["code"] == "job_queue_full"
        assert response.json()["error"]["details"]["max_queued_jobs"] == 1
        await wait_for_status(client, str(first["id"]), {"completed"})


async def test_job_creation_rejects_upload_over_size_limit() -> None:
    app = create_app(settings=Settings(max_upload_mb=1))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/v1/audio/transcription-jobs",
            files={"file": ("large.wav", b"x" * (1024 * 1024 + 1), "audio/wav")},
        )

    assert response.status_code == 413
    body = response.json()
    assert body["error"]["code"] == "audio_too_large"
    assert body["error"]["details"]["max_upload_mb"] == 1


async def test_sync_transcription_over_duration_threshold_returns_job(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "SYNC_JOB_THRESHOLD_SECONDS", 0.01)
    async def probe_duration(_path: Path, **_kwargs: object) -> float:
        return 0.1

    monkeypatch.setattr(main_module, "probe_audio_duration_path_async", probe_duration)

    def fail_decode(audio: bytes) -> object:
        del audio
        raise AssertionError("sync duration threshold should not require full decode before returning 202")

    monkeypatch.setattr(main_module, "run_transcription_path", fail_decode)

    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["status_url"] == f"/v1/jobs/{body['id']}"
    await wait_for_status(client, str(body["id"]), {"completed"})


async def test_expired_job_is_removed_and_returns_not_found() -> None:
    app = create_app(settings=Settings(job_result_ttl_seconds=1))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        created = await create_job(client)
        await wait_for_status(client, str(created["id"]), {"completed"})
        job = app.state.job_manager._jobs[str(created["id"])]
        job.expires_at = utc_now() - timedelta(seconds=1)

        response = await client.get(f"/v1/jobs/{created['id']}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "job_not_found"
    assert str(created["id"]) not in app.state.job_manager._jobs


async def test_shutdown_cancels_queued_jobs_instead_of_draining_queue() -> None:
    app = create_app(settings=Settings(job_shutdown_grace_seconds=1), adapter_delay_seconds=0.1)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        first = await create_job(client)
        await wait_for_status(client, str(first["id"]), {"preprocessing", "splitting", "loading_model", "transcribing"})
        second = await create_job(client)

        await app.state.job_manager.shutdown()

        second_job = app.state.job_manager._jobs[str(second["id"])]
        assert second_job.status == "cancelled"
        assert not second_job.work_dir.exists()
