from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from asr_mcp.client import AsrHttpClient, BridgeSettings
from asr_mcp.errors import AsrGatewayError


async def test_transcribe_file_uploads_allowed_audio_to_existing_gateway(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFF-test-audio")
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["authorization"] = request.headers.get("Authorization")
        captured["content_type"] = request.headers.get("Content-Type")
        captured["body"] = request.content
        return httpx.Response(
            200,
            json={
                "id": "tr_test",
                "model": "qwen3-asr-1.7b",
                "text": "测试转录",
            },
        )

    settings = BridgeSettings(
        base_url="http://asr.test:18080",
        allowed_roots=(tmp_path,),
        bearer_token="secret-token",
        max_file_mb=1,
        request_timeout_seconds=30.0,
    )
    client = AsrHttpClient(settings, transport=httpx.MockTransport(handler))

    result = await client.transcribe_file(
        audio_path,
        model="qwen3-asr-1.7b",
        language="zh",
        context="项目术语",
    )

    assert result["text"] == "测试转录"
    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/audio/transcriptions"
    assert captured["authorization"] == "Bearer secret-token"
    assert str(captured["content_type"]).startswith("multipart/form-data; boundary=")
    captured_body = captured["body"]
    assert isinstance(captured_body, bytes)
    assert b"sample.wav" in captured_body
    assert b"RIFF-test-audio" in captured_body
    assert b"qwen3-asr-1.7b" in captured_body
    assert b"project" not in captured_body
    assert "项目术语".encode() in captured_body


async def test_transcribe_file_rejects_audio_outside_allowed_roots(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside_audio = tmp_path / "private.wav"
    outside_audio.write_bytes(b"private-audio")
    request_started = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal request_started
        request_started = True
        return httpx.Response(200, json={"text": "unexpected"})

    client = AsrHttpClient(
        BridgeSettings(
            base_url="http://asr.test:18080",
            allowed_roots=(allowed_root,),
            max_file_mb=1,
        ),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ValueError, match="ASR_MCP_ALLOWED_ROOTS"):
        await client.transcribe_file(outside_audio)

    assert request_started is False


def test_bridge_settings_loads_mac_client_configuration_from_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "audio"
    second_root = tmp_path / "meetings"
    monkeypatch.setenv("ASR_MCP_BASE_URL", "http://192.168.31.137:18080/")
    monkeypatch.setenv("ASR_MCP_ALLOWED_ROOTS", f"{first_root}:{second_root}")
    monkeypatch.setenv("ASR_MCP_TOKEN", "local-secret")
    monkeypatch.setenv("ASR_MCP_MAX_FILE_MB", "256")
    monkeypatch.setenv("ASR_MCP_REQUEST_TIMEOUT_SECONDS", "1800")

    settings = BridgeSettings.from_env()

    assert settings.base_url == "http://192.168.31.137:18080"
    assert settings.allowed_roots == (first_root, second_root)
    assert settings.bearer_token == "local-secret"
    assert settings.max_file_mb == 256
    assert settings.request_timeout_seconds == 1800.0


def test_bridge_settings_rejects_filesystem_root_as_allowed_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASR_MCP_ALLOWED_ROOTS", "/")

    with pytest.raises(ValueError, match="filesystem root"):
        BridgeSettings.from_env()


async def test_list_models_returns_gateway_capability_discovery(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "id": "qwen3-asr-1.7b",
                        "status": "loaded",
                        "capabilities": {"backends": ["transformers"]},
                    }
                ]
            },
        )

    client = AsrHttpClient(
        BridgeSettings(base_url="http://asr.test:18080", allowed_roots=(tmp_path,)),
        transport=httpx.MockTransport(handler),
    )

    result = await client.list_models()

    assert result["models"][0]["id"] == "qwen3-asr-1.7b"
    assert result["models"][0]["capabilities"]["backends"] == ["transformers"]


async def test_create_job_forwards_long_audio_options(tmp_path: Path) -> None:
    audio_path = tmp_path / "meeting.mp3"
    audio_path.write_bytes(b"long-audio")
    body = b""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal body
        body = request.content
        assert request.method == "POST"
        assert request.url.path == "/v1/audio/transcription-jobs"
        return httpx.Response(
            202,
            json={"id": "job_test", "status": "queued", "queue_position": 1},
        )

    client = AsrHttpClient(
        BridgeSettings(base_url="http://asr.test:18080", allowed_roots=(tmp_path,)),
        transport=httpx.MockTransport(handler),
    )

    result = await client.create_job(
        audio_path,
        model="moss-transcribe-diarize-0.9b",
        response_format="verbose_json",
        hotwords=["Qwen3-ASR", "Anchor Replay"],
        split_strategy="fixed",
        max_chunk_seconds=1200,
        preserve_segments=True,
        speaker_resolution="auto",
    )

    assert result == {"id": "job_test", "status": "queued", "queue_position": 1}
    assert b"moss-transcribe-diarize-0.9b" in body
    assert b"verbose_json" in body
    assert b"Qwen3-ASR" in body
    assert b"Anchor Replay" in body
    assert b"1200" in body
    assert b"true" in body


async def test_job_can_be_inspected_and_cancelled_through_existing_routes(tmp_path: Path) -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"id": "job_123", "status": "transcribing", "progress": {"percent": 50.0}},
            )
        return httpx.Response(200, json={"id": "job_123", "status": "cancel_requested"})

    client = AsrHttpClient(
        BridgeSettings(base_url="http://asr.test:18080", allowed_roots=(tmp_path,)),
        transport=httpx.MockTransport(handler),
    )

    current = await client.get_job("job_123")
    cancelled = await client.cancel_job("job_123")

    assert current["progress"] == {"percent": 50.0}
    assert cancelled["status"] == "cancel_requested"
    assert requests == [
        ("GET", "/v1/jobs/job_123"),
        ("DELETE", "/v1/jobs/job_123"),
    ]


async def test_gateway_error_envelope_is_preserved_for_mcp_callers(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            request=request,
            json={
                "error": {
                    "code": "capability_not_supported",
                    "message": "timestamps are not supported",
                    "details": {"timestamps": "word"},
                }
            },
        )

    client = AsrHttpClient(
        BridgeSettings(base_url="http://asr.test:18080", allowed_roots=(tmp_path,)),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(AsrGatewayError) as raised:
        await client.list_models()

    assert raised.value.status_code == 422
    assert raised.value.code == "capability_not_supported"
    assert raised.value.details == {"timestamps": "word"}
    assert str(raised.value) == "capability_not_supported: timestamps are not supported"


async def test_model_lifecycle_operations_use_declared_management_routes(tmp_path: Path) -> None:
    requests: list[tuple[str, str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path, request.content))
        if request.method == "GET":
            return httpx.Response(200, json={"id": "qwen3-asr-1.7b", "status": "unloaded"})
        if request.method == "POST":
            return httpx.Response(200, json={"id": "qwen3-asr-1.7b", "status": "loading"})
        return httpx.Response(200, json={"id": "qwen3-asr-1.7b", "status": "unloading_scheduled"})

    client = AsrHttpClient(
        BridgeSettings(base_url="http://asr.test:18080", allowed_roots=(tmp_path,)),
        transport=httpx.MockTransport(handler),
    )

    status = await client.get_model_status("qwen3-asr-1.7b")
    loading = await client.load_model("qwen3-asr-1.7b", backend="transformers")
    unloading = await client.unload_model("qwen3-asr-1.7b")

    assert status["status"] == "unloaded"
    assert loading["status"] == "loading"
    assert unloading["status"] == "unloading_scheduled"
    assert [(method, path) for method, path, _body in requests] == [
        ("GET", "/v1/models/qwen3-asr-1.7b/status"),
        ("POST", "/v1/models/qwen3-asr-1.7b/load"),
        ("DELETE", "/v1/models/qwen3-asr-1.7b"),
    ]
    assert b'"backend":"transformers"' in requests[1][2]
    assert b'"mode":"after_current_requests"' in requests[2][2]
