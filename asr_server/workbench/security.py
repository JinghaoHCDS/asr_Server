from __future__ import annotations

import hashlib
import hmac
import secrets
from urllib.parse import urlsplit

from fastapi import Request

from asr_server.errors import AsrError


COOKIE_NAME = "asr_workbench_session"


def password_hash(password: str) -> str:
    if len(password) < 8 or len(password) > 1024:
        raise AsrError(400, "bad_request", "password must contain 8 to 1024 characters")
    salt = secrets.token_hex(16)
    key = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1)
    return f"scrypt${salt}${key.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, salt, expected = stored.split("$")
        if algorithm != "scrypt" or len(password) > 1024:
            return False
        actual = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1)
        return hmac.compare_digest(actual.hex(), expected)
    except (ValueError, TypeError):
        return False


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def check_same_origin(request: Request) -> None:
    """Reject browser cross-site mutations, including login CSRF."""
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    if request.headers.get("sec-fetch-site") == "cross-site":
        raise AsrError(403, "cross_origin_request", "cross-site requests are not allowed")
    origin = request.headers.get("origin")
    if origin is not None:
        parsed = urlsplit(origin)
        actual = urlsplit(str(request.url))
        if (parsed.scheme, parsed.hostname, parsed.port) != (actual.scheme, actual.hostname, actual.port):
            raise AsrError(403, "cross_origin_request", "request origin does not match this server")
