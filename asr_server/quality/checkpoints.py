from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from asr_server.adapters.base import ContextApplication, GenerationMetrics, TimestampItem, TranscriptionResult, TranscriptionSegment, TranscriptionTimings
from asr_server.errors import AsrError


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


class Checkpoints:
    def __init__(self, directory: Path | None, identity: str) -> None:
        self.directory = directory
        self.identity = identity
        self.extra_seconds = 0.0
        self.review_attempts: dict[str, int] = {}
        if directory is None:
            return
        metadata = directory / "identity.json"
        if metadata.exists():
            if json.loads(metadata.read_text())["fingerprint"] != identity:
                raise AsrError(409, "checkpoint_version_mismatch", "stored model, workflow, glossary or input differs; start a new task")
        else:
            atomic_json(metadata, {"fingerprint": identity})
        ledger = directory / "review-budget.json"
        if ledger.exists():
            stored = json.loads(ledger.read_text())
            self.extra_seconds = float(stored["extra_seconds"])
            self.review_attempts = {str(key): int(value) for key, value in stored.get("attempts", {}).items()}

    def reserve_review(self, seconds: float, budget: float, key: str = "") -> None:
        if self.extra_seconds + seconds > budget + 0.001:
            raise AsrError(422, "review_budget_exhausted", "automatic acoustic review budget is exhausted")
        self.extra_seconds += seconds
        if key:
            if self.review_attempts.get(key, 0) >= 2:
                self.extra_seconds -= seconds
                raise AsrError(422, "review_budget_exhausted", "automatic review attempt limit is exhausted")
            self.review_attempts[key] = self.review_attempts.get(key, 0) + 1
        # Reserve before invoking the worker so crashes cannot reset cost.
        if self.directory is not None:
            atomic_json(self.directory / "review-budget.json", {"extra_seconds": self.extra_seconds, "attempts": self.review_attempts})

    def load(self, index: int) -> tuple[TranscriptionResult, list[dict[str, object]], float] | None:
        if self.directory is None:
            return None
        path = self.directory / f"chunk-{index:05d}.json"
        if not path.exists():
            return None
        data: dict[str, Any] = json.loads(path.read_text())
        raw = data["result"]
        result = TranscriptionResult(
            text=raw["text"], duration=raw["duration"], language=raw["language"], warnings=raw["warnings"],
            segments=[TranscriptionSegment(**item) for item in raw["segments"]],
            timestamps=[TimestampItem(**item) for item in raw["timestamps"]],
            timings=TranscriptionTimings(**raw["timings"]), generation=GenerationMetrics(**raw["generation"]),
            context=ContextApplication(**raw["context"]),
        )
        return result, data.get("reviews", []), float(data.get("extra_seconds", 0))

    def save(self, index: int, result: TranscriptionResult, reviews: list[dict[str, object]], extra_seconds: float) -> None:
        if self.directory is not None:
            atomic_json(self.directory / f"chunk-{index:05d}.json", {"result": asdict(result), "reviews": reviews, "extra_seconds": extra_seconds})
