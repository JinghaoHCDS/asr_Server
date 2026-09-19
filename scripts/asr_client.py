from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx


DEFAULT_BASE_URL = "http://192.168.31.137:18080"
DEFAULT_MODEL = "qwen3-asr-1.7b"
FALLBACK_MODEL = "qwen3-asr-0.6b"


def make_client(timeout_seconds: float) -> httpx.Client:
    timeout = httpx.Timeout(connect=5.0, read=timeout_seconds, write=30.0, pool=5.0)
    return httpx.Client(timeout=timeout, trust_env=False)


def response_error_code(response: httpx.Response) -> str | None:
    try:
        body: dict[str, Any] = response.json()
    except ValueError:
        return None
    error = body.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    return code if isinstance(code, str) else None


def discover_models(client: httpx.Client, base_url: str) -> list[dict[str, Any]]:
    response = client.get(f"{base_url}/v1/models")
    response.raise_for_status()
    body = response.json()
    models = body.get("models")
    if not isinstance(models, list):
        raise RuntimeError("/v1/models response did not contain a models list")
    return [model for model in models if isinstance(model, dict)]


def choose_fallback_model(models: list[dict[str, Any]], current_model: str) -> str | None:
    for preferred in (FALLBACK_MODEL, DEFAULT_MODEL):
        if preferred != current_model and any(model.get("id") == preferred for model in models):
            return preferred
    for model in models:
        model_id = model.get("id")
        capabilities = model.get("capabilities")
        if (
            isinstance(model_id, str)
            and model_id != current_model
            and isinstance(capabilities, dict)
            and capabilities.get("transcription") is True
        ):
            return model_id
    return None


def declared_backends(models: list[dict[str, Any]], model_id: str) -> list[str]:
    for model in models:
        if model.get("id") != model_id:
            continue
        capabilities = model.get("capabilities")
        if not isinstance(capabilities, dict):
            return []
        backends = capabilities.get("backends")
        if not isinstance(backends, list):
            return []
        return [backend for backend in backends if isinstance(backend, str)]
    return []


def choose_backend(models: list[dict[str, Any]], model_id: str, requested_backend: str) -> str:
    if requested_backend == "auto":
        return "auto"
    backends = declared_backends(models, model_id)
    if requested_backend in backends:
        return requested_backend
    if backends:
        return backends[0]
    return "auto"


def choose_model_for_purpose(
    models: list[dict[str, Any]],
    purpose: str,
    requested_model: str | None,
) -> str | None:
    if requested_model is not None:
        candidates = [model for model in models if model.get("id") == requested_model]
    else:
        candidates = models
    for model in candidates:
        capabilities = model.get("capabilities")
        if not isinstance(capabilities, dict):
            continue
        if purpose == "oral_analysis" and not (
            capabilities.get("forced_alignment") is True and capabilities.get("timestamps")
        ):
            continue
        if purpose == "podcast" and not (
            capabilities.get("diarization") is True
            and "auto" in capabilities.get("speaker_resolution_modes", [])
        ):
            continue
        model_id = model.get("id")
        if isinstance(model_id, str):
            return model_id
    if requested_model is not None:
        raise RuntimeError(f"model {requested_model!r} cannot satisfy purpose={purpose!r}")
    return None


def check_server(base_url: str) -> None:
    with make_client(timeout_seconds=30.0) as client:
        health = client.get(f"{base_url}/health")
        health.raise_for_status()
        models = client.get(f"{base_url}/v1/models")
        models.raise_for_status()

    print("health:")
    print(health.text)
    print("models:")
    print(models.text)


def post_transcription(
    client: httpx.Client,
    base_url: str,
    audio_path: Path,
    *,
    model: str | None,
    backend: str,
    language: str,
    timestamps: str | None,
    purpose: str,
    context: str,
    hotwords: str | None,
    max_new_tokens: int | None,
) -> httpx.Response:
    data = {
        "backend": backend,
        "language": language,
        "purpose": purpose,
    }
    if model:
        data["model"] = model
    if timestamps:
        data["timestamps"] = timestamps
    if context:
        data["context"] = context
    if hotwords:
        data["hotwords"] = hotwords
    if max_new_tokens is not None:
        data["max_new_tokens"] = str(max_new_tokens)
    with audio_path.open("rb") as audio_file:
        return client.post(
            f"{base_url}/v1/audio/transcriptions",
            files={"file": (audio_path.name, audio_file, "application/octet-stream")},
            data=data,
        )


def transcribe(
    base_url: str,
    audio_path: Path,
    model: str | None,
    backend: str,
    language: str,
    timestamps: str | None,
    purpose: str = "transcription",
    context: str = "",
    hotwords: str | None = None,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    with make_client(timeout_seconds=1800.0) as client:
        models = discover_models(client, base_url)
        current_model = choose_model_for_purpose(models, purpose, model)
        if current_model is None and purpose != "transcription":
            raise RuntimeError(f"no runtime model can satisfy purpose={purpose!r}")
        current_model = current_model or model or DEFAULT_MODEL
        current_backend = choose_backend(models, current_model, backend)
        current_timestamps = timestamps

        for attempt in range(1, 22):
            response = post_transcription(
                client,
                base_url,
                audio_path,
                model=current_model,
                backend=current_backend,
                language=language,
                timestamps=current_timestamps,
                purpose=purpose,
                context=context,
                hotwords=hotwords,
                max_new_tokens=max_new_tokens,
            )
            if response.status_code == 200:
                result = response.json()
                if not isinstance(result, dict):
                    raise RuntimeError("transcription response was not a JSON object")
                return result

            code = response_error_code(response)
            if response.status_code == 409 and code == "model_loading" and attempt <= 20:
                time.sleep(3)
                continue
            if response.status_code == 409 and code == "model_unloading_scheduled":
                fallback = choose_fallback_model(models, current_model)
                if fallback is not None:
                    current_model = fallback
                    current_backend = choose_backend(models, current_model, backend)
                    continue
            response.raise_for_status()

    raise RuntimeError("transcription retries exhausted")


def create_job_and_wait(
    base_url: str,
    audio_path: Path,
    *,
    model: str | None,
    backend: str,
    language: str,
    timestamps: str | None,
    purpose: str,
    context: str,
    hotwords: str | None,
    max_new_tokens: int | None,
    wait: bool,
    poll_seconds: float,
) -> dict[str, Any]:
    with make_client(timeout_seconds=1800.0) as client:
        models = discover_models(client, base_url)
        selected_model = choose_model_for_purpose(models, purpose, model)
        if selected_model is None and purpose != "transcription":
            raise RuntimeError(f"no runtime model can satisfy purpose={purpose!r}")
        selected_model = selected_model or model or DEFAULT_MODEL
        data: dict[str, str] = {
            "model": selected_model,
            "backend": choose_backend(models, selected_model, backend),
            "language": language,
            "purpose": purpose,
        }
        if timestamps:
            data["timestamps"] = timestamps
        if context:
            data["context"] = context
        if hotwords:
            data["hotwords"] = hotwords
        if max_new_tokens is not None:
            data["max_new_tokens"] = str(max_new_tokens)
        with audio_path.open("rb") as audio_file:
            response = client.post(
                f"{base_url}/v1/audio/transcription-jobs",
                files={"file": (audio_path.name, audio_file, "application/octet-stream")},
                data=data,
            )
        response.raise_for_status()
        created = response.json()
        if not isinstance(created, dict):
            raise RuntimeError("job creation response was not a JSON object")
        if not wait:
            return created
        job_id = str(created["id"])
        while True:
            status_response = client.get(
                f"{base_url}/v1/jobs/{job_id}",
                params={"detail": "compact"},
            )
            status_response.raise_for_status()
            status = status_response.json()
            if status.get("status") in {"completed", "failed", "cancelled"}:
                full_response = client.get(
                    f"{base_url}/v1/jobs/{job_id}",
                    params={"detail": "full"},
                )
                full_response.raise_for_status()
                full = full_response.json()
                if not isinstance(full, dict):
                    raise RuntimeError("job status response was not a JSON object")
                return full
            time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mac 侧 ASR 服务验证客户端。")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="检查 /health 和 /v1/models，且禁用环境代理。")

    transcribe_parser = subparsers.add_parser("transcribe", help="上传一个音频文件进行转录。")
    transcribe_parser.add_argument("audio_path", type=Path)
    transcribe_parser.add_argument("--model")
    transcribe_parser.add_argument("--backend", default="auto", choices=["auto", "transformers"])
    transcribe_parser.add_argument("--language", default="auto")
    transcribe_parser.add_argument("--timestamps", choices=["none", "auto", "word", "char"])
    transcribe_parser.add_argument(
        "--purpose",
        default="transcription",
        choices=["transcription", "oral_analysis", "podcast"],
    )
    transcribe_parser.add_argument("--context", default="")
    transcribe_parser.add_argument("--hotwords", default=None)
    transcribe_parser.add_argument("--max-new-tokens", type=int, default=None)

    job_parser = subparsers.add_parser("job", help="创建异步任务，可在本地低频轮询到完成。")
    job_parser.add_argument("audio_path", type=Path)
    job_parser.add_argument("--model")
    job_parser.add_argument("--backend", default="auto", choices=["auto", "transformers"])
    job_parser.add_argument("--language", default="auto")
    job_parser.add_argument("--timestamps", choices=["none", "auto", "word", "char"])
    job_parser.add_argument(
        "--purpose",
        required=True,
        choices=["transcription", "oral_analysis", "podcast"],
    )
    job_parser.add_argument("--context", default="")
    job_parser.add_argument("--hotwords")
    job_parser.add_argument("--max-new-tokens", type=int)
    job_parser.add_argument("--no-wait", action="store_true")
    job_parser.add_argument("--poll-seconds", type=float, default=10.0)

    args = parser.parse_args()
    if args.command == "check":
        check_server(args.base_url)
        return

    if args.command == "job":
        result = create_job_and_wait(
            args.base_url,
            args.audio_path,
            model=args.model,
            backend=args.backend,
            language=args.language,
            timestamps=args.timestamps,
            purpose=args.purpose,
            context=args.context,
            hotwords=args.hotwords,
            max_new_tokens=args.max_new_tokens,
            wait=not args.no_wait,
            poll_seconds=args.poll_seconds,
        )
    else:
        result = transcribe(
            args.base_url,
            args.audio_path,
            args.model,
            args.backend,
            args.language,
            args.timestamps,
            purpose=args.purpose,
            context=args.context,
            hotwords=args.hotwords,
            max_new_tokens=args.max_new_tokens,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
