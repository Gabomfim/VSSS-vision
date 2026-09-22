"""Reproducible input, runtime, source-control, and artifact provenance."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import platform
import subprocess
import sys
import math

import cv2
import numpy as np


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path, role: str | None = None) -> dict:
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    record = {
        "path": str(source),
        "filename": source.name,
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "sha256": sha256_file(source),
    }
    if role is not None:
        record["role"] = role
    return record


def input_record(value: int | str, role: str) -> dict:
    if isinstance(value, str) and Path(value).expanduser().is_file():
        return file_record(value, role)
    return {"role": role, "kind": "camera_or_stream", "identifier": value}


def _git_record(working_directory: str | Path | None = None) -> dict:
    cwd = str(working_directory or Path.cwd())

    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    root = run("rev-parse", "--show-toplevel")
    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    diff = run("diff", "--binary", "HEAD")
    return {
        "repository_root": root,
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "tracked_diff_sha256": (
            sha256(diff.encode("utf-8")).hexdigest() if diff else None
        ),
    }


def runtime_record(working_directory: str | Path | None = None) -> dict:
    return {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "working_directory": str(Path.cwd().resolve()),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "git": _git_record(working_directory),
    }


def write_manifest(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    temporary.replace(destination)


def artifact_records(paths: list[str | Path]) -> list[dict]:
    return [file_record(path, "generated_artifact") for path in paths]


def json_safe(value):
    """Convert NumPy scalars and non-finite floats into strict JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
