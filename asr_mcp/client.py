from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from asr_mcp.errors import AsrGatewayError


@dataclass(frozen=True)
class BridgeSettings:
    base_url: str
    allowed_roots: tuple[Path, ...]
    bearer_token: str | None = None
    max_file_mb: int = 512
    request_timeout_seconds: float = 3600.0

    def __post_init__(self) -> None:
        if not self.allowed_roots:
            raise ValueError("ASR_MCP_ALLOWED_ROOTS must contain at least one local directory")
        filesystem_root = Path(Path.home().anchor).resolve()
        home_root = Path.home().resolve()
        for root in self.allowed_roots:
            resolved = root.expanduser().resolve()
            if resolved == filesystem_root:
                raise ValueError("ASR_MCP_ALLOWED_ROOTS cannot include the filesystem root")
            if resolved == home_root:
                raise ValueError("ASR_MCP_ALLOWED_ROOTS cannot include the entire home directory")
        if self.max_file_mb < 1:
            raise ValueError("ASR_MCP_MAX_FILE_MB must be greater than or equal to 1")
        if self.request_timeout_seconds <= 0:
            raise ValueError("ASR_MCP_REQUEST_TIMEOUT_SECONDS must be greater than 0")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    @classmethod
    def from_env(cls) -> BridgeSettings:
        roots_value = os.getenv("ASR_MCP_ALLOWED_ROOTS", "")
        allowed_roots = tuple(
            Path(value.strip()).expanduser()
            for value in roots_value.split(os.pathsep)
            if value.strip()
        )
        max_file_mb = int(os.getenv("ASR_MCP_MAX_FILE_MB", "512"))
        request_timeout_seconds = float(os.getenv("ASR_MCP_REQUEST_TIMEOUT_SECONDS", "3600"))
        return cls(
            base_url=os.getenv("ASR_MCP_BASE_URL", "http://192.168.31.137:18080"),
            allowed_roots=allowed_roots,
            bearer_token=os.getenv("ASR_MCP_TOKEN") or None,
            max_file_mb=max_file_mb,
            request_timeout_seconds=request_timeout_seconds,
        )


class AsrHttpClient:
    def __init__(
        self,
        settings: BridgeSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    async def list_models(self) -> dict[str, Any]:
        return await self._request_json("GET", "/v1/models")

    async def get_model_status(self, model_id: str) -> dict[str, Any]:
        return await self._request_json(
            "GET",
            f"/v1/models/{quote(model_id, safe='')}/status",
        )

    async def load_model(
        self,
        model_id: str,
        *,
        backend: str = "auto",
        device: str = "cuda",
        dtype: str = "auto",
    ) -> dict[str, Any]:
        return await self._request_json(
            "POST",
            f"/v1/models/{quote(model_id, safe='')}/load",
            json={"backend": backend, "device": device, "dtype": dtype},
        )

    async def unload_model(
        self,
        model_id: str,
        *,
        cuda_empty_cache: bool = True,
    ) -> dict[str, Any]:
        return await self._request_json(
            "DELETE",
            f"/v1/models/{quote(model_id, safe='')}",
            json={
                "mode": "after_current_requests",
                "reject_new_requests": True,
                "cuda_empty_cache": cuda_empty_cache,
            },
        )

    async def get_job(self, job_id: str, *, detail: str = "compact") -> dict[str, Any]:
        return await self._request_json(
            "GET",
            f"/v1/jobs/{quote(job_id, safe='')}",
            params={"detail": detail},
        )

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        return await self._request_json("DELETE", f"/v1/jobs/{quote(job_id, safe='')}")

    async def create_job(
        self,
        audio_path: Path,
        *,
        model: str | None = None,
        language: str = "auto",
        response_format: str | None = None,
        timestamps: str | None = None,
        backend: str = "auto",
        max_new_tokens: int | None = None,
        context: str = "",
        hotwords: list[str] | None = None,
        split_strategy: str = "auto",
        max_chunk_seconds: float | None = None,
        overlap_seconds: float | None = None,
        preserve_segments: bool | None = None,
        speaker_resolution: str | None = None,
        purpose: str = "transcription",
        pause_threshold_seconds: float = 0.8,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        path = self.validate_local_audio_path(audio_path)
        data = self._transcription_form(
            model=model,
            language=language,
            response_format=response_format,
            timestamps=timestamps,
            backend=backend,
            max_new_tokens=max_new_tokens,
            context=context,
            hotwords=hotwords,
            split_strategy=split_strategy,
            max_chunk_seconds=max_chunk_seconds,
            overlap_seconds=overlap_seconds,
            preserve_segments=preserve_segments,
            speaker_resolution=speaker_resolution,
            purpose=purpose,
            pause_threshold_seconds=pause_threshold_seconds,
            idempotency_key=idempotency_key,
        )
        with path.open("rb") as audio_file:
            response = await self._request(
                "POST",
                "/v1/audio/transcription-jobs",
                files={"file": (path.name, audio_file, "application/octet-stream")},
                data=data,
            )
        return self._response_json(response)

    async def transcribe_file(
        self,
        audio_path: Path,
        *,
        model: str | None = None,
        language: str = "auto",
        response_format: str | None = None,
        timestamps: str | None = None,
        backend: str = "auto",
        max_new_tokens: int | None = None,
        context: str = "",
        hotwords: list[str] | None = None,
        split_strategy: str = "auto",
        max_chunk_seconds: float | None = None,
        overlap_seconds: float | None = None,
        preserve_segments: bool | None = None,
        speaker_resolution: str | None = None,
        purpose: str = "transcription",
        pause_threshold_seconds: float = 0.8,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        path = self.validate_local_audio_path(audio_path)
        data = self._transcription_form(
            model=model,
            language=language,
            response_format=response_format,
            timestamps=timestamps,
            backend=backend,
            max_new_tokens=max_new_tokens,
            context=context,
            hotwords=hotwords,
            split_strategy=split_strategy,
            max_chunk_seconds=max_chunk_seconds,
            overlap_seconds=overlap_seconds,
            preserve_segments=preserve_segments,
            speaker_resolution=speaker_resolution,
            purpose=purpose,
            pause_threshold_seconds=pause_threshold_seconds,
            idempotency_key=idempotency_key,
        )
        with path.open("rb") as audio_file:
            response = await self._request(
                "POST",
                "/v1/audio/transcriptions",
                files={"file": (path.name, audio_file, "application/octet-stream")},
                data=data,
            )
        return self._response_json(response)

    def validate_local_audio_path(self, audio_path: Path) -> Path:
        try:
            resolved = audio_path.expanduser().resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"audio file does not exist: {audio_path}") from exc
        if not resolved.is_file():
            raise ValueError(f"audio path is not a regular file: {resolved}")
        roots = tuple(root.expanduser().resolve() for root in self._settings.allowed_roots)
        if not roots or not any(resolved.is_relative_to(root) for root in roots):
            raise ValueError("audio file is outside ASR_MCP_ALLOWED_ROOTS")
        max_bytes = self._settings.max_file_mb * 1024 * 1024
        if resolved.stat().st_size > max_bytes:
            raise ValueError(f"audio file exceeds MCP bridge limit of {self._settings.max_file_mb} MB")
        return resolved

    async def _request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = await self._request(method, path, **kwargs)
        return self._response_json(response)

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        headers = {}
        if self._settings.bearer_token:
            headers["Authorization"] = f"Bearer {self._settings.bearer_token}"
        timeout = httpx.Timeout(
            connect=5.0,
            read=self._settings.request_timeout_seconds,
            write=30.0,
            pool=5.0,
        )
        async with httpx.AsyncClient(
            base_url=self._settings.base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            trust_env=False,
            transport=self._transport,
        ) as client:
            return await client.request(method, path, **kwargs)

    @staticmethod
    def _transcription_form(
        *,
        model: str | None,
        language: str,
        response_format: str | None,
        timestamps: str | None,
        backend: str,
        max_new_tokens: int | None,
        context: str,
        hotwords: list[str] | None,
        split_strategy: str,
        max_chunk_seconds: float | None,
        overlap_seconds: float | None,
        preserve_segments: bool | None,
        speaker_resolution: str | None,
        purpose: str,
        pause_threshold_seconds: float,
        idempotency_key: str | None,
    ) -> dict[str, str]:
        data = {
            "language": language,
            "backend": backend,
            "context": context,
            "split_strategy": split_strategy,
            "purpose": purpose,
            "pause_threshold_seconds": str(pause_threshold_seconds),
        }
        if response_format is not None:
            data["response_format"] = response_format
        if timestamps is not None:
            data["timestamps"] = timestamps
        if preserve_segments is not None:
            data["preserve_segments"] = str(preserve_segments).lower()
        if speaker_resolution is not None:
            data["speaker_resolution"] = speaker_resolution
        if model is not None:
            data["model"] = model
        if max_new_tokens is not None:
            data["max_new_tokens"] = str(max_new_tokens)
        if hotwords:
            data["hotwords"] = json.dumps(hotwords, ensure_ascii=False)
        if max_chunk_seconds is not None:
            data["max_chunk_seconds"] = str(max_chunk_seconds)
        if overlap_seconds is not None:
            data["overlap_seconds"] = str(overlap_seconds)
        if idempotency_key is not None:
            data["idempotency_key"] = idempotency_key
        return data

    @staticmethod
    def _response_json(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError:
            response.raise_for_status()
            raise RuntimeError("ASR gateway returned invalid JSON")
        if response.is_error:
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                code = error.get("code")
                message = error.get("message")
                details = error.get("details")
                if isinstance(code, str) and isinstance(message, str):
                    raise AsrGatewayError(
                        status_code=response.status_code,
                        code=code,
                        message=message,
                        details=details if isinstance(details, dict) else {},
                    )
            response.raise_for_status()
        if not isinstance(payload, dict):
            raise RuntimeError("ASR gateway returned a non-object JSON response")
        return payload
