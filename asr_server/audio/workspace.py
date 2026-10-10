from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import UploadFile

from asr_server.config import Settings
from asr_server.errors import AsrError


UPLOAD_BLOCK_BYTES = 1024 * 1024


@dataclass
class AudioWorkspace:
    root: Path
    upload_path: Path
    size_bytes: int
    _manager: WorkspaceManager
    _cleaned: bool = False
    _cleanup_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

    async def cleanup(self) -> None:
        if self._cleaned:
            return
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup())
        cleanup_task = self._cleanup_task
        # All callers join the same deletion and budget release, including
        # cancelled callers; terminal job states must not precede file cleanup.
        cancelled = False
        while True:
            try:
                await asyncio.shield(cleanup_task)
                break
            except asyncio.CancelledError:
                if cleanup_task.cancelled():
                    raise
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    async def _cleanup(self) -> None:
        await asyncio.to_thread(shutil.rmtree, self.root, True)
        await self._manager.release(self.size_bytes)
        self._cleaned = True


class WorkspaceManager:
    """Own request/job workspaces and a process-wide upload spool budget."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = asyncio.Lock()
        self._reserved_upload_bytes = 0

    async def store_upload(self, upload: UploadFile) -> AudioWorkspace:
        root = Path(tempfile.mkdtemp(prefix="asr_job_"))
        (root / "owner.pid").write_text(str(os.getpid()), encoding="ascii")
        upload_path = root / "upload"
        total = 0
        try:
            with upload_path.open("wb") as output:
                while True:
                    block = await upload.read(UPLOAD_BLOCK_BYTES)
                    if not block:
                        break
                    await self._reserve(len(block), root)
                    total += len(block)
                    try:
                        await asyncio.to_thread(output.write, block)
                    except Exception:
                        await self.release(len(block))
                        total -= len(block)
                        raise
            return AudioWorkspace(root=root, upload_path=upload_path, size_bytes=total, _manager=self)
        except Exception:
            await asyncio.to_thread(shutil.rmtree, root, True)
            await self.release(total)
            raise

    async def release(self, size_bytes: int) -> None:
        async with self._lock:
            self._reserved_upload_bytes = max(self._reserved_upload_bytes - size_bytes, 0)

    async def _reserve(self, size_bytes: int, root: Path) -> None:
        max_upload_bytes = self._settings.max_upload_mb * 1024 * 1024
        max_spool_bytes = self._settings.max_spool_mb * 1024 * 1024
        async with self._lock:
            projected = self._reserved_upload_bytes + size_bytes
            current_workspace_bytes = 0
            upload_path = root / "upload"
            if upload_path.exists():
                current_workspace_bytes = upload_path.stat().st_size
            if current_workspace_bytes + size_bytes > max_upload_bytes:
                raise AsrError(
                    413,
                    "audio_too_large",
                    "audio upload exceeds the server size limit",
                    {"max_upload_bytes": max_upload_bytes, "max_upload_mb": self._settings.max_upload_mb},
                )
            if projected > max_spool_bytes:
                raise AsrError(
                    429,
                    "job_queue_full",
                    "audio spool byte budget is exhausted",
                    {"max_spool_bytes": max_spool_bytes},
                )
            free_bytes = shutil.disk_usage(root).free
            min_free_bytes = self._settings.min_free_disk_mb * 1024 * 1024
            if free_bytes - size_bytes < min_free_bytes:
                raise AsrError(
                    503,
                    "storage_unavailable",
                    "insufficient temporary disk space",
                    {"free_bytes": free_bytes, "min_free_disk_bytes": min_free_bytes},
                )
            self._reserved_upload_bytes = projected


def validate_workspace_limits(root: Path, settings: Settings) -> None:
    files = [path for path in root.rglob("*") if path.is_file()]
    if len(files) > settings.max_workspace_files:
        raise AsrError(
            503,
            "storage_unavailable",
            "audio workspace file count limit exceeded",
            {"file_count": len(files), "max_workspace_files": settings.max_workspace_files},
        )
    size_bytes = sum(path.stat().st_size for path in files)
    max_bytes = settings.max_workspace_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise AsrError(
            503,
            "storage_unavailable",
            "audio workspace byte limit exceeded",
            {"workspace_bytes": size_bytes, "max_workspace_bytes": max_bytes},
        )
