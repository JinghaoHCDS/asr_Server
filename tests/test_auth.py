from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from asr_server.auth import BearerAuthMiddleware
from asr_server.config import Settings, load_settings
from asr_server.main import create_app
from starlette.types import Message, Scope


@pytest.mark.parametrize("authorization", [None, "Bearer wrong", "Basic test-secret", "Bearer"])
async def test_api_rejects_missing_or_invalid_credentials(authorization: str | None) -> None:
    app = create_app(Settings(api_token="test-secret"))
    headers = {"Authorization": authorization} if authorization is not None else {}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers=headers) as client:
        response = await client.post("/v1/audio/transcription-jobs", content=b"invalid multipart")
        assert response.status_code == 401
        assert response.json() == {
            "error": {"code": "unauthorized", "message": "a valid Bearer token is required", "details": {}}
        }
        assert response.headers["www-authenticate"] == "Bearer"
        assert (await client.get("/v1/models")).status_code == 401
        assert (await client.delete("/v1/models")).status_code == 401


async def test_valid_token_allows_models_and_lifecycle_operations() -> None:
    app = create_app(Settings(api_token="test-secret"))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
        headers={"Authorization": "bearer test-secret"},
    ) as client:
        assert (await client.get("/v1/models")).status_code == 200
        assert (await client.post("/v1/models/qwen3-asr-0.6b/load")).status_code == 200
        assert (await client.delete("/v1/models/qwen3-asr-0.6b")).status_code == 200


async def test_health_and_docs_stay_public_and_swagger_exposes_authorization() -> None:
    app = create_app(Settings(api_token="test-secret"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/docs")).status_code == 200
        schema = (await client.get("/openapi.json")).json()
    assert schema["components"]["securitySchemes"]["BearerAuth"]["scheme"] == "bearer"
    assert schema["paths"]["/v1/models"]["get"]["security"] == [{"BearerAuth": []}]
    assert "security" not in schema["paths"]["/health"]["get"]


async def test_disabled_auth_preserves_existing_api_behavior() -> None:
    app = create_app(Settings())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/v1/models")).status_code == 200


async def test_rejected_upload_never_reads_body_or_calls_application() -> None:
    async def unexpected_app(scope: Scope, receive: object, send: object) -> None:
        raise AssertionError("unauthenticated request reached the application")

    async def unexpected_receive() -> Message:
        raise AssertionError("unauthenticated upload body was consumed")

    messages: list[Message] = []

    async def send(message: Message) -> None:
        messages.append(message)

    middleware = BearerAuthMiddleware(unexpected_app, token="test-secret")
    await middleware(
        {"type": "http", "path": "/v1/audio/transcriptions", "headers": [], "method": "POST"},
        unexpected_receive, send,
    )
    assert messages[0]["status"] == 401


def test_token_configuration_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_API_TOKEN", "test-secret")
    assert load_settings().api_token == "test-secret"
    monkeypatch.setenv("ASR_API_TOKEN", "")
    assert load_settings().api_token is None
