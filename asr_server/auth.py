from __future__ import annotations

import hmac
from typing import Any

from fastapi import FastAPI

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from asr_server.errors import error_payload


class GatewayApp(FastAPI):
    def openapi(self) -> dict[str, Any]:
        schema = super().openapi()
        if self.state.settings.api_token:
            schema.setdefault("components", {}).setdefault("securitySchemes", {})["BearerAuth"] = {
                "type": "http", "scheme": "bearer"
            }
            for path, operations in schema["paths"].items():
                if path.startswith("/v1/"):
                    for method, operation in operations.items():
                        if method in {"get", "post", "put", "patch", "delete", "head", "options"}:
                            operation["security"] = [{"BearerAuth": []}]
        return schema


class BearerAuthMiddleware:
    """Check optional API credentials before reading an uploaded request body."""

    def __init__(self, app: ASGIApp, *, token: str) -> None:
        self.app = app
        self.token = token.encode("utf-8")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        if scope["type"] == "http" and (path == "/v1" or path.startswith("/v1/")):
            authorization = Headers(scope=scope).get("authorization", "")
            scheme, _, credential = authorization.partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(
                credential.encode("utf-8"), self.token
            ):
                response = JSONResponse(
                    status_code=401,
                    content=error_payload("unauthorized", "a valid Bearer token is required"),
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
