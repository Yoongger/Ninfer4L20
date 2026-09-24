"""Shared helpers for the Ninfer4L20 port tooling.

Standard library only. Everything here is deliberately environment-agnostic: the port engine
works on any checkout of the upstream tree as long as `git` and `python3` exist, so the same
code runs on the bare Ubuntu host, inside the WSL distro, and inside the Docker build image.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def log_step(message: str) -> None:
    print(f"\n########## {message} ##########", flush=True)


def log_ok(message: str) -> None:
    print(f"  [ ok ] {message}", flush=True)


def log_warn(message: str) -> None:
    print(f"  [warn] {message}", flush=True)


def log_fail(message: str) -> None:
    print(f"  [FAIL] {message}", flush=True)


def load_manifest(path: Path | None = None) -> dict:
    manifest_path = path or (REPO_ROOT / "port" / "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def git(worktree: Path, *args: str, check: bool = True, capture: bool = True,
        config: list[tuple[str, str]] | None = None) -> subprocess.CompletedProcess:
    """Run a git command in the worktree. stderr is captured too so diagnostics stay attached.
    `config` carries -c key=value pairs (they must precede the subcommand)."""
    prefix: list[str] = ["git", "-C", str(worktree)]
    for key, value in (config or []):
        prefix += ["-c", f"{key}={value}"]
    cmd = prefix + list(args)
    proc = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"git {' '.join(args)} failed ({proc.returncode}): {detail}")
    return proc


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return handle.read()


def read_file_lines(path: Path):
    """Return (text, crlf) with LF-normalised text. crlf is True when the file uses CRLF, so the
    caller can restore the original line ending on write (a Windows/WSL checkout may have CRLF)."""
    raw = read_text(path)
    crlf = "\r\n" in raw
    return (raw.replace("\r\n", "\n") if crlf else raw), crlf


def write_text(path: Path, text: str, crlf: bool) -> None:
    if crlf:
        text = text.replace("\n", "\r\n")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def sha_of(text: str) -> str:
    """Blob SHA-1 for content, matching `git hash-object` (so index lines stay resolvable)."""
    import hashlib

    body = text.encode("utf-8")
    store = b"blob " + str(len(body)).encode("ascii") + b"\0" + body
    return hashlib.sha1(store).hexdigest()


def parse_index_line(patch_text: str):
    """Extract (base_sha, result_sha) from the first `index` line of a unified diff, if present."""
    match = re.search(r"^index ([0-9a-f]+)\.\.([0-9a-f]+)", patch_text, re.MULTILINE)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def patch_target_relpath(patch_text: str) -> str | None:
    """The a/... path of the first file section in a (single-file) unified diff."""
    match = re.search(r"^diff --git a/(\S+) b/", patch_text, re.MULTILINE)
    if not match:
        return None
    return match.group(1)


def version_at_least(version: str, minimum: str) -> bool:
    """Dotted-version comparison that tolerates trailing junk ('13.1.2-devel' etc.)."""

    def parts(value: str):
        out = []
        for chunk in value.split("."):
            digits = re.match(r"^\d+", chunk)
            out.append(int(digits.group(0)) if digits else 0)
        return out

    a, b = parts(version), parts(minimum)
    length = max(len(a), len(b))
    a += [0] * (length - len(a))
    b += [0] * (length - len(b))
    return a >= b


def die(message: str, code: int = 1) -> None:
    log_fail(message)
    sys.exit(code)
