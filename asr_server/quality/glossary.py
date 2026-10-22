from __future__ import annotations

import re
from typing import Any

from asr_server.context import parse_hotwords, prepare_context


def entries(snapshot: dict[str, object] | None) -> list[dict[str, Any]]:
    raw = (snapshot or {}).get("entries", [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict) and isinstance(item.get("canonical"), str)]


def glossary_context(context: str, hotwords: str | None, snapshot: dict[str, object] | None) -> tuple[str, dict[str, object]]:
    requested = parse_hotwords(hotwords)
    requested.extend(str(item["canonical"]) for item in entries(snapshot))
    # Determine the actually adopted prefix, including the existing global
    # prompt budget; prepare_context is the shared API normalization contract.
    import json

    encoded = json.dumps(requested, ensure_ascii=False)
    prepared = prepare_context(context, encoded)
    requested = parse_hotwords(encoded)
    used = requested[:prepared.hotword_count]
    return prepared.prompt, {
        "version": (snapshot or {}).get("version"),
        "used": used,
        "omitted": requested[len(used):],
        "truncated": prepared.truncated,
        "warnings": list(prepared.warnings),
        "context": {**prepared.request_summary(), "applied": bool(prepared.prompt), "method": "qwen3_asr_system_prompt" if prepared.prompt else None, "ignored_reason": None},
    }


def normalize_terms(text: str, snapshot: dict[str, object] | None) -> tuple[str, list[dict[str, object]], list[str]]:
    """Apply only explicit, unambiguous, administrator-approved aliases.

    Case/word boundaries keep e.g. 'ASR' from replacing a substring of an
    unrelated Latin word. Colliding canonical/alias spellings are never auto
    corrected, irrespective of dictionary ordering.
    """
    candidates: dict[str, set[str]] = {}
    allowed: dict[tuple[str, str], bool] = {}
    for item in entries(snapshot):
        canonical = str(item["canonical"]).strip()
        candidates.setdefault(canonical.casefold(), set()).add(canonical)
        aliases = item.get("aliases", [])
        if not isinstance(aliases, list):
            continue
        for alias in aliases:
            if not isinstance(alias, str) or not alias.strip() or alias == canonical:
                continue
            alias = alias.strip()
            candidates.setdefault(alias.casefold(), set()).add(canonical)
            allowed[(alias.casefold(), canonical)] = bool(item.get("auto_normalize", False))
    changes: list[dict[str, object]] = []
    ambiguous: list[str] = []
    # A single regex pass prevents one replacement becoming a second alias.
    aliases = [key for key in candidates if any(pair[0] == key for pair in allowed)]
    if not aliases:
        return text, changes, ambiguous
    alternatives = []
    for alias in sorted(aliases, key=len, reverse=True):
        left = r"(?<![A-Za-z0-9_])" if alias[0].isascii() and alias[0].isalnum() else ""
        right = r"(?![A-Za-z0-9_])" if alias[-1].isascii() and alias[-1].isalnum() else ""
        alternatives.append(left + re.escape(alias) + right)
    pattern = re.compile("|".join(alternatives), re.IGNORECASE)

    def replace(match: re.Match[str]) -> str:
        original = match.group()
        key = original.casefold()
        targets = candidates[key]
        canonical = next(iter(targets))
        if len(targets) != 1 or not allowed.get((key, canonical), False):
            ambiguous.append(original)
            return original
        if original == canonical:
            return original
        changes.append({"start_char": match.start(), "end_char": match.end(), "before": original, "after": canonical, "method": "confirmed_alias"})
        return canonical

    return pattern.sub(replace, text), changes, list(dict.fromkeys(ambiguous))
