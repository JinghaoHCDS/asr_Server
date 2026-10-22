from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, cast
from uuid import uuid4

from asr_server.errors import AsrError
from asr_server.workbench.security import password_hash, token_hash


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def after_days(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def identifier(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class WorkbenchStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "workbench.sqlite3"
        self.initialized = False

    def initialize(self) -> None:
        if self.initialized:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL, is_admin INTEGER NOT NULL,
                    disabled INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS uploads (
                    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, filename TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL, offset INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, expires_at TEXT NOT NULL, job_id TEXT
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE, owner_id TEXT NOT NULL,
                    status TEXT NOT NULL, reserved_bytes INTEGER NOT NULL,
                    payload TEXT NOT NULL, idempotency_key TEXT, fingerprint TEXT,
                    UNIQUE(owner_id,idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS versions (
                    job_id TEXT NOT NULL, version INTEGER NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(job_id,version)
                );
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS jobs_owner ON jobs(owner_id,sequence);
            """)
            db.execute("INSERT OR IGNORE INTO metadata VALUES (?,?)", ("glossary", json.dumps({
                "version": 1,
                "entries": [
                    {"canonical": "Qwen3-ASR", "aliases": [], "domain": "ai", "auto_normalize": False},
                    {"canonical": "ChatGPT", "aliases": ["chatgpt"], "domain": "ai", "auto_normalize": True},
                    {"canonical": "OpenAI", "aliases": ["openai"], "domain": "ai", "auto_normalize": True},
                    {"canonical": "API", "aliases": [], "domain": "general", "auto_normalize": False},
                ],
            }, ensure_ascii=False)))
        self.initialized = True

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self.initialize()
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def create_user(self, username: str, password: str, is_admin: bool = False) -> dict[str, Any]:
        username = username.strip()
        if not re.fullmatch(r"[\w.@-]{2,64}", username):
            raise AsrError(400, "bad_request", "username must contain 2 to 64 letters, digits, . @ - or _")
        hashed = password_hash(password)
        user = {"id": identifier("user"), "username": username, "is_admin": is_admin,
                "disabled": False, "created_at": now()}
        try:
            with self.connection() as db:
                db.execute("INSERT INTO users VALUES (?,?,?,?,?,?)", (
                    user["id"], username, hashed, int(is_admin), 0, user["created_at"],
                ))
        except sqlite3.IntegrityError as exc:
            raise AsrError(409, "username_exists", "username already exists") from exc
        return user

    @staticmethod
    def public_user(row: sqlite3.Row) -> dict[str, Any]:
        return {"id": row["id"], "username": row["username"], "is_admin": bool(row["is_admin"]),
                "disabled": bool(row["disabled"]), "created_at": row["created_at"]}

    def authenticate_session(self, token: str) -> dict[str, Any]:
        with self.connection() as db:
            row = db.execute("SELECT u.* FROM users u JOIN sessions s ON u.id=s.user_id "
                             "WHERE s.token_hash=? AND s.expires_at>? AND u.disabled=0", (token_hash(token), now())).fetchone()
        if row is None:
            raise AsrError(401, "unauthorized", "sign in to the workbench")
        return self.public_user(row)

    def jobs(self, owner_id: str | None = None) -> list[dict[str, Any]]:
        with self.connection() as db:
            if owner_id is None:
                rows = db.execute("SELECT payload FROM jobs ORDER BY sequence").fetchall()
            else:
                rows = db.execute("SELECT payload FROM jobs WHERE owner_id=? ORDER BY sequence DESC", (owner_id,)).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def job(self, job_id: str, owner_id: str | None = None) -> dict[str, Any]:
        with self.connection() as db:
            if owner_id is None:
                row = db.execute("SELECT payload FROM jobs WHERE id=?", (job_id,)).fetchone()
            else:
                row = db.execute("SELECT payload FROM jobs WHERE id=? AND owner_id=?", (job_id, owner_id)).fetchone()
        if row is None:
            raise AsrError(404, "job_not_found", "workbench job not found")
        return cast(dict[str, Any], json.loads(row["payload"]))

    def save_job(self, job: dict[str, Any], reserved_bytes: int | None = None) -> None:
        with self.connection() as db:
            db.execute("UPDATE jobs SET status=?,payload=? WHERE id=?", (job["status"], json.dumps(job, ensure_ascii=False), job["id"]))
            if reserved_bytes is not None:
                db.execute("UPDATE jobs SET reserved_bytes=? WHERE id=?", (reserved_bytes, job["id"]))

    def save_version(self, job: dict[str, Any]) -> None:
        result = job["result"]
        with self.connection() as db:
            db.execute("INSERT OR REPLACE INTO versions VALUES (?,?,?)", (
                job["id"], result["version"], json.dumps(result, ensure_ascii=False),
            ))
            db.execute("UPDATE jobs SET status=?,payload=? WHERE id=?", (job["status"], json.dumps(job, ensure_ascii=False), job["id"]))

    def version(self, job_id: str, version: int) -> dict[str, Any]:
        with self.connection() as db:
            row = db.execute("SELECT payload FROM versions WHERE job_id=? AND version=?", (job_id, version)).fetchone()
        if row is None:
            raise AsrError(404, "version_not_found", "transcript version not found")
        return cast(dict[str, Any], json.loads(row["payload"]))

    def usage(self, owner_id: str | None = None) -> int:
        clause = " WHERE owner_id=?" if owner_id is not None else ""
        args = (owner_id,) if owner_id is not None else ()
        with self.connection() as db:
            jobs = db.execute("SELECT coalesce(sum(reserved_bytes),0) FROM jobs" + clause, args).fetchone()[0]
            metadata = db.execute("SELECT coalesce(sum(length(CAST(payload AS BLOB))),0) FROM jobs" + clause, args).fetchone()[0]
            versions_clause = " WHERE j.owner_id=?" if owner_id is not None else ""
            versions = db.execute("SELECT coalesce(sum(length(CAST(v.payload AS BLOB))),0) FROM versions v JOIN jobs j ON j.id=v.job_id" + versions_clause, args).fetchone()[0]
            upload_clause = " WHERE job_id IS NULL" + (" AND owner_id=?" if owner_id is not None else "")
            uploads = db.execute("SELECT coalesce(sum(size_bytes),0) FROM uploads" + upload_clause, args).fetchone()[0]
        return int(jobs) + int(uploads) + int(metadata) + int(versions)

    def glossary(self) -> dict[str, Any]:
        with self.connection() as db:
            row = db.execute("SELECT payload FROM metadata WHERE key='glossary'").fetchone()
        assert row is not None
        return cast(dict[str, Any], json.loads(row[0]))
