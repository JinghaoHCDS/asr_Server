from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx


DEFAULT_BASE_URL = "http://192.168.31.137:18080"
TERMINAL = {"completed", "failed", "cancelled"}
_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


class BatchGateway(Protocol):
    def models(self) -> list[dict[str, Any]]: ...

    def submit(self, path: Path, fields: dict[str, str]) -> dict[str, Any]: ...

    def job(self, job_id: str, *, detail: str) -> dict[str, Any]: ...


class HttpBatchGateway:
    def __init__(self, base_url: str, token: str | None = None) -> None:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(connect=5, read=30, write=1800, pool=5),
            trust_env=False,
        )

    def close(self) -> None:
        self._client.close()

    def models(self) -> list[dict[str, Any]]:
        response = self._client.get("/v1/models")
        response.raise_for_status()
        models = response.json().get("models")
        if not isinstance(models, list):
            raise RuntimeError("/v1/models did not return a models list")
        return [item for item in models if isinstance(item, dict)]

    def submit(self, path: Path, fields: dict[str, str]) -> dict[str, Any]:
        with path.open("rb") as source:
            response = self._client.post(
                "/v1/audio/transcription-jobs",
                data=fields,
                files={"file": (path.name, source, "application/octet-stream")},
            )
        return _checked_json(response)

    def job(self, job_id: str, *, detail: str) -> dict[str, Any]:
        response = self._client.get(f"/v1/jobs/{job_id}", params={"detail": detail})
        return _checked_json(response)


@dataclass(frozen=True)
class PreparedItem:
    item_id: str
    path: Path
    content_sha256: str
    metadata: dict[str, Any]
    fields: dict[str, str]
    cache_key: str


class BatchRunner:
    def __init__(
        self,
        gateway: BatchGateway,
        manifest_path: Path,
        output_dir: Path,
        *,
        max_in_flight: int = 4,
        max_attempts: int = 2,
        poll_seconds: float = 3.0,
        retry_failed: bool = False,
    ) -> None:
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be at least 1")
        self.gateway = gateway
        self.manifest_path = manifest_path.resolve()
        self.output_dir = output_dir.resolve()
        self.max_in_flight = max_in_flight
        self.max_attempts = max_attempts
        self.poll_seconds = poll_seconds
        self.retry_failed = retry_failed
        self.models = gateway.models()
        self.items = self._prepare_items()
        self.state_path = self.output_dir / "manifest.json"
        self.state = self._load_state()

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        while True:
            self._reconcile_cache()
            active = [entry for entry in self.state["items"] if entry.get("status") in {"submitted", "running"}]
            pending = [entry for entry in self.state["items"] if entry.get("status") in {"pending", "retry"}]
            while pending and len(active) < self.max_in_flight:
                entry = pending.pop(0)
                prepared = self.items[str(entry["id"])]
                try:
                    response = self.gateway.submit(prepared.path, self._submission_fields(entry, prepared))
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 429:
                        break
                    self._submission_failure(entry, exc)
                    continue
                except httpx.TransportError as exc:
                    # The server may have accepted the upload before the client
                    # lost the response. Keep the same idempotency key so this
                    # retry cannot create a second computation in that case.
                    self._submission_failure(entry, exc)
                    continue
                entry.update(
                    status="submitted",
                    job_id=response["id"],
                    attempts=int(entry.get("attempts", 0)) + 1,
                    last_error=None,
                )
                active.append(entry)
                self._save_state()

            for entry in list(active):
                job_id = str(entry["job_id"])
                try:
                    job = self.gateway.job(job_id, detail="compact")
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 404:
                        self._retry_or_fail(entry, "job_not_found after server restart or expiry")
                        self._save_state()
                        continue
                    raise
                except httpx.TransportError as exc:
                    entry["last_error"] = f"temporary poll failure: {exc}"
                    self._save_state()
                    continue
                status = str(job.get("status"))
                entry["status"] = "running" if status not in TERMINAL else status
                entry["server_status"] = status
                entry["progress"] = job.get("progress")
                if status == "completed":
                    try:
                        full = self.gateway.job(job_id, detail="full")
                    except httpx.TransportError as exc:
                        entry["status"] = "running"
                        entry["last_error"] = f"temporary final-result failure: {exc}"
                        self._save_state()
                        continue
                    except httpx.HTTPStatusError as exc:
                        if exc.response.status_code == 404:
                            self._retry_or_fail(entry, "job expired before final result retrieval")
                            self._save_state()
                            continue
                        raise
                    result = full.get("result")
                    if not isinstance(result, dict):
                        self._retry_or_fail(
                            entry,
                            "completed job did not expose a full result",
                            new_computation=True,
                        )
                    else:
                        self._write_result(entry, result)
                elif status in {"failed", "cancelled"}:
                    entry["last_error"] = job.get("error") or {"code": status, "message": status}
                    self._retry_or_fail(
                        entry,
                        json.dumps(entry["last_error"], ensure_ascii=False),
                        new_computation=True,
                    )
                self._save_state()

            unfinished = [
                entry
                for entry in self.state["items"]
                if entry.get("status") in {"pending", "retry", "submitted", "running"}
            ]
            if not unfinished:
                self._write_index()
                return self.state
            self._write_index()
            if self.poll_seconds > 0:
                time.sleep(self.poll_seconds)

    def _prepare_items(self) -> dict[str, PreparedItem]:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        raw_items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(raw_items, list):
            raise ValueError("batch manifest must contain an items array")
        prepared: dict[str, PreparedItem] = {}
        for raw in raw_items:
            if not isinstance(raw, dict):
                raise ValueError("every manifest item must be an object")
            path_value = raw.get("path")
            if not isinstance(path_value, str):
                raise ValueError("every manifest item requires path")
            path = Path(path_value).expanduser()
            if not path.is_absolute():
                path = self.manifest_path.parent / path
            path = path.resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"media path is not a file: {path}")
            content_sha = _file_sha256(path)
            raw_id = str(raw.get("id") or path.stem)
            item_id = _stable_item_id(raw_id, content_sha)
            if item_id in prepared:
                raise ValueError(f"duplicate stable item id: {item_id}")
            purpose = str(raw.get("purpose", "transcription"))
            model = _choose_model(self.models, purpose, raw.get("model"))
            model_id = str(model["id"])
            fields = {
                "model": model_id,
                "backend": str(raw.get("backend", "auto")),
                "language": str(raw.get("language", "auto")),
                "purpose": purpose,
            }
            context = raw.get("context")
            if isinstance(context, str) and context:
                fields["context"] = context
            keywords = raw.get("keywords", raw.get("hotwords"))
            if isinstance(keywords, list) and all(isinstance(value, str) for value in keywords):
                fields["hotwords"] = json.dumps(keywords, ensure_ascii=False)
            for key in (
                "timestamps",
                "response_format",
                "split_strategy",
                "speaker_resolution",
                "max_new_tokens",
                "max_chunk_seconds",
                "overlap_seconds",
                "pause_threshold_seconds",
            ):
                if raw.get(key) is not None:
                    fields[key] = str(raw[key]).lower() if isinstance(raw[key], bool) else str(raw[key])
            capabilities = model.get("capabilities") if isinstance(model.get("capabilities"), dict) else {}
            canonical = {
                "content_sha256": content_sha,
                "model": model_id,
                "model_revision": model.get("revision"),
                "capabilities": capabilities,
                "fields": fields,
            }
            cache_key = hashlib.sha256(
                json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest()
            fields["idempotency_key"] = f"batch:{cache_key}"
            prepared[item_id] = PreparedItem(
                item_id=item_id,
                path=path,
                content_sha256=content_sha,
                metadata={key: raw.get(key) for key in ("title", "source", "description") if raw.get(key) is not None},
                fields=fields,
                cache_key=cache_key,
            )
        return prepared

    def _load_state(self) -> dict[str, Any]:
        previous: dict[str, Any] = {}
        if self.state_path.exists():
            loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous = {
                    str(item.get("id")): item
                    for item in loaded.get("items", [])
                    if isinstance(item, dict) and item.get("id") is not None
                }
        entries = []
        for item_id, prepared in self.items.items():
            old = previous.get(item_id, {})
            same = old.get("cache_key") == prepared.cache_key
            previous_status = old.get("status", "pending") if same else "pending"
            if self.retry_failed and previous_status in {"failed", "cancelled"}:
                previous_status = "retry"
            previous_generation = int(old.get("submission_generation", 0)) if same else 0
            if self.retry_failed and previous_status == "retry":
                previous_generation += 1
            entries.append(
                {
                    "id": item_id,
                    "path": str(prepared.path),
                    "content_sha256": prepared.content_sha256,
                    "cache_key": prepared.cache_key,
                    "parameters": prepared.fields,
                    "metadata": prepared.metadata,
                    "status": previous_status,
                    "job_id": old.get("job_id") if same else None,
                    "attempts": 0 if self.retry_failed and previous_status == "retry" else old.get("attempts", 0) if same else 0,
                    "last_error": old.get("last_error") if same else None,
                    "result_dir": old.get("result_dir") if same else None,
                    "submission_generation": previous_generation,
                }
            )
        return {"schema_version": 1, "source_manifest": str(self.manifest_path), "items": entries}

    def _reconcile_cache(self) -> None:
        for entry in self.state["items"]:
            if entry.get("status") != "completed":
                continue
            result_dir = self.output_dir / "items" / str(entry["id"])
            if not (result_dir / "transcript.json").is_file() or not (result_dir / "transcript.md").is_file():
                entry["status"] = "retry"

    def _submission_failure(self, entry: dict[str, Any], exc: Exception) -> None:
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        self._retry_or_fail(entry, str(exc))
        self._save_state()

    def _retry_or_fail(
        self,
        entry: dict[str, Any],
        message: str,
        *,
        new_computation: bool = False,
    ) -> None:
        entry["last_error"] = message
        entry["job_id"] = None
        if new_computation:
            entry["submission_generation"] = int(entry.get("submission_generation", 0)) + 1
        entry["status"] = "retry" if int(entry.get("attempts", 0)) < self.max_attempts else "failed"

    def _submission_fields(self, entry: dict[str, Any], prepared: PreparedItem) -> dict[str, str]:
        fields = dict(prepared.fields)
        generation = int(entry.get("submission_generation", 0))
        if generation > 0:
            fields["idempotency_key"] = f"batch:{prepared.cache_key}:compute-{generation}"
        return fields

    def _write_result(self, entry: dict[str, Any], result: dict[str, Any]) -> None:
        item_id = str(entry["id"])
        prepared = self.items[item_id]
        target = self.output_dir / "items" / item_id
        target.mkdir(parents=True, exist_ok=True)
        artifact = {
            "schema_version": 1,
            "sample": False,
            "status": "completed",
            "item": {
                "id": item_id,
                "path": str(prepared.path),
                "content_sha256": prepared.content_sha256,
                **prepared.metadata,
            },
            "request": prepared.fields,
            "cache_key": prepared.cache_key,
            "result": result,
        }
        _atomic_json(target / "transcript.json", artifact)
        _atomic_text(target / "transcript.md", _result_markdown(artifact))
        entry["status"] = "completed"
        entry["result_dir"] = str(target.relative_to(self.output_dir))
        entry["last_error"] = None

    def _save_state(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.state_path, self.state)
        self._write_index()

    def _write_index(self) -> None:
        lines = ["# ASR batch output", "", "This index is generated from manifest.json.", ""]
        for entry in self.state["items"]:
            item_id = str(entry["id"])
            status = str(entry.get("status"))
            result_link = f"items/{item_id}/transcript.md" if status == "completed" else "-"
            title = entry.get("metadata", {}).get("title") or Path(str(entry["path"])).name
            rendered_result = f"[{result_link}]({result_link})" if status == "completed" else "-"
            lines.append(f"- [{status}] {title} (`{item_id}`): {rendered_result}")
        _atomic_text(self.output_dir / "index.md", "\n".join(lines) + "\n")


def _choose_model(models: list[dict[str, Any]], purpose: str, requested: object) -> dict[str, Any]:
    candidates = [model for model in models if requested is None or model.get("id") == requested]
    if purpose == "oral_analysis":
        candidates = [
            model
            for model in candidates
            if isinstance(model.get("capabilities"), dict)
            and bool(model["capabilities"].get("forced_alignment"))
            and bool(model["capabilities"].get("timestamps"))
        ]
    elif purpose == "podcast":
        candidates = [
            model
            for model in candidates
            if isinstance(model.get("capabilities"), dict)
            and bool(model["capabilities"].get("diarization"))
            and "auto" in model["capabilities"].get("speaker_resolution_modes", [])
        ]
    else:
        candidates = [
            model
            for model in candidates
            if isinstance(model.get("capabilities"), dict) and model["capabilities"].get("transcription") is True
        ]
    if not candidates:
        raise ValueError(f"no runtime model can satisfy purpose={purpose!r} requested_model={requested!r}")
    return next((model for model in candidates if model.get("default") is True), candidates[0])


def _result_markdown(artifact: dict[str, Any]) -> str:
    item = artifact["item"]
    result = artifact["result"]
    purpose = artifact["request"].get("purpose", "transcription")
    lines = [f"# {item.get('title') or Path(item['path']).name}", ""]
    lines.extend(
        [
            f"- item_id: {item['id']}",
            f"- source: {item.get('source', '')}",
            f"- purpose: {purpose}",
            f"- model: {result.get('model')}",
            f"- duration_seconds: {result.get('duration')}",
            f"- context_applied: {result.get('context', {}).get('applied')}",
            f"- warnings: {json.dumps(result.get('warnings', []), ensure_ascii=False)}",
            "",
        ]
    )
    if purpose == "oral_analysis":
        lines.extend(["## Time-located transcript", ""])
        reading = result.get("timing", {}).get("reading_segments", [])
        for segment in reading if isinstance(reading, list) else []:
            lines.append(f"[{_clock(segment['start'])} – {_clock(segment['end'])}] {segment['text']}")
            lines.append("")
    elif purpose == "podcast":
        lines.extend(["## Speaker transcript", ""])
        for segment in result.get("segments", []):
            speaker = segment.get("speaker") or "unknown"
            lines.append(f"**{speaker}:** {segment.get('text', '')}")
            lines.append("")
    else:
        lines.extend(["## Transcript", "", str(result.get("text", "")), ""])
    return "\n".join(lines).rstrip() + "\n"


def _clock(value: object) -> str:
    if not isinstance(value, int | float | str):
        raise TypeError(f"expected timestamp value, got {type(value).__name__}")
    seconds = max(float(value), 0.0)
    whole = int(seconds)
    return f"{whole // 3600:02d}:{(whole % 3600) // 60:02d}:{whole % 60:02d}.{int((seconds - whole) * 1000):03d}"


def _stable_item_id(raw: str, content_sha: str) -> str:
    safe = _SAFE_ID.sub("-", raw).strip("-._")[:80] or "item"
    return f"{safe}-{content_sha[:12]}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _checked_json(response: httpx.Response) -> dict[str, Any]:
    payload = response.json()
    if response.is_error:
        response.raise_for_status()
    if not isinstance(payload, dict):
        raise RuntimeError("ASR gateway returned a non-object response")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit, resume, and export an ASR batch manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--token", default=os.getenv("ASR_TOKEN"))
    parser.add_argument("--max-in-flight", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gateway = HttpBatchGateway(args.base_url, token=args.token)
    try:
        state = BatchRunner(
            gateway,
            args.manifest,
            args.output,
            max_in_flight=args.max_in_flight,
            max_attempts=args.max_attempts,
            poll_seconds=args.poll_seconds,
            retry_failed=args.retry_failed,
        ).run()
    finally:
        gateway.close()
    completed = sum(item.get("status") == "completed" for item in state["items"])
    failed = sum(item.get("status") == "failed" for item in state["items"])
    print(json.dumps({"completed": completed, "failed": failed, "output": str(args.output)}, ensure_ascii=False))
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
