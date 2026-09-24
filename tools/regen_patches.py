#!/usr/bin/env python3
"""Re-record the per-change patch files against the current upstream base.

The patch files in port/changes/ are derived artifacts: they are the diff between the upstream
base and the ported tree, one file per change. Because they are derived, they can always be
re-based onto a newer upstream tip - which is what keeps them in the state where a plain 2-way
`git apply` succeeds, so the three-way and transform ladders stay as fallbacks rather than the
main path.

Usage (run after the port has been applied to the worktree, e.g. after scripts/upgrade.sh):
    python3 tools/regen_patches.py --worktree PATH [--base REF]

The default base is HEAD~1 when HEAD is a port commit, otherwise HEAD (the worktree must then
carry the port as uncommitted changes; pass --base explicitly if the layout differs).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    REPO_ROOT,
    die,
    git,
    load_manifest,
    log_ok,
    log_step,
    log_warn,
    sha_of,
    write_text,
)


def file_at(worktree: Path, ref: str, relpath: str) -> str:
    proc = git(worktree, "show", f"{ref}:{relpath}", check=False)
    if proc.returncode != 0:
        return None
    text = proc.stdout
    return text.replace("\r\n", "\n")


def unified_diff(relpath: str, old: str, new: str, old_sha: str, new_sha: str) -> str:
    import difflib

    old_lines = old.split("\n")
    new_lines = new.split("\n")
    lines = [
        f"diff --git a/{relpath} b/{relpath}",
        f"index {old_sha[:12]}..{new_sha[:12]} 100644",
        f"--- a/{relpath}",
        f"+++ b/{relpath}",
    ]
    hunk = list(difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{relpath}",
                                     tofile=f"+++ b/{relpath}".replace("+++ b/", f"+++ b/"),
                                     lineterm=""))
    # difflib emits the header itself with our fromfile/tofile; keep only the hunks.
    body = [line for line in hunk if not line.startswith(("--- ", "+++ ", "diff ", "index "))]
    lines.extend(body)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", default=str(REPO_ROOT / "src"))
    parser.add_argument("--base", default=None,
                        help="upstream ref the patches are recorded against (default: auto)")
    args = parser.parse_args()

    worktree = Path(args.worktree).resolve()
    if not (worktree / "CMakeLists.txt").exists():
        die(f"worktree not found: {worktree}")
    manifest = load_manifest()

    base = args.base
    if base is None:
        # If HEAD is a port commit, record the patches against its parent (the upstream base);
        # otherwise the worktree must carry the port as uncommitted changes and the base is HEAD.
        head_subject = git(worktree, "log", "-1", "--format=%s").stdout.strip()
        prefix = manifest["layout"]["port_commit_prefix"]
        has_parent = git(worktree, "rev-parse", "--verify", "-q", "HEAD~1", check=False).returncode == 0
        if head_subject.startswith(prefix) and has_parent:
            base = "HEAD~1"
        else:
            base = "HEAD"
    base_sha = git(worktree, "rev-parse", base).stdout.strip()
    log_step(f"regenerating change patches against {base} ({base_sha[:12]})")

    for change in manifest["changes"]:
        if change.get("kind") != "patch":
            continue
        relpath = change["file"]
        out_path = REPO_ROOT / change["patch"]
        old = file_at(worktree, base, relpath)
        new = (worktree / relpath).read_text(encoding="utf-8", errors="replace")
        new = new.replace("\r\n", "\n")
        if old is None:
            log_warn(f"{change['id']}: file not in base ref, skipping ({relpath})")
            continue
        if old == new:
            log_warn(f"{change['id']}: no difference between base and worktree for {relpath}")
            continue
        old_sha = sha_of(old)
        new_sha = sha_of(new)
        # Prefer the real blob SHAs when the base file is a tracked blob in the object store.
        real_old = git(worktree, "rev-parse", f"{base}:{relpath}", check=False).stdout.strip()
        if re.fullmatch(r"[0-9a-f]{40}", real_old or ""):
            old_sha = real_old
        patch_text = unified_diff(relpath, old, new, old_sha, new_sha)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_text(out_path, patch_text, crlf=False)
        log_ok(f"{change['id']}: {out_path.relative_to(REPO_ROOT)} "
               f"(base {old_sha[:12]} -> worktree {new_sha[:12]})")

    log_step("done")
    log_ok("the recorded patches now 2-way-apply at the current base; commit them with the port repo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
