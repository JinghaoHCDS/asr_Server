from __future__ import annotations

import re
import tomllib
from pathlib import Path

from asr_server.adapters.moss import MODEL_REVISIONS as MOSS_REVISIONS
from asr_server.adapters.qwen import MODEL_REVISIONS as QWEN_REVISIONS


FULL_COMMIT = re.compile(r"@[0-9a-f]{40}$")


def test_gpu_git_dependencies_are_pinned_to_full_commits() -> None:
    lines = Path("requirements/wsl-gpu-cu128.txt").read_text(encoding="utf-8").splitlines()
    git_lines = [line for line in lines if "git+https://" in line]

    assert git_lines
    assert all(FULL_COMMIT.search(line) for line in git_lines)


def test_hugging_face_model_revisions_are_full_snapshot_ids() -> None:
    revisions = [*QWEN_REVISIONS.values(), *MOSS_REVISIONS.values()]

    assert revisions
    assert all(re.fullmatch(r"[0-9a-f]{40}", revision) for revision in revisions)


def test_mcp_bridge_stays_on_stable_v1_sdk_line() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    optional_dependencies = pyproject["project"]["optional-dependencies"]["mcp"]
    dev_dependencies = pyproject["dependency-groups"]["dev"]

    assert "mcp>=1.27,<2" in optional_dependencies
    assert "mcp>=1.27,<2" in dev_dependencies
