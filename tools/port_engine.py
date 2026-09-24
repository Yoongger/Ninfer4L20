#!/usr/bin/env python3
"""Apply the ninfer-l20 port to a checkout of the upstream tree.

Design goals (this is the rewrite):
  * No pinned commits. The port is applied to whatever upstream commit the worktree is on;
    provenance is recorded afterwards, never consumed as an input.
  * Every change is applied by a ladder of strategies, per change:
        1. direct 2-way `git apply` (works whenever upstream context survived),
        2. explicit three-way merge (`git merge-file` on base/current/theirs),
        3. declarative regex transforms (a "variant ladder" of known source shapes, so a
           refactored upstream structure is still recognised and rewritten),
        4. a structured drift report (what was expected, what was found, what to do next).
    A change never fails the whole run silently; the verify layer is the final gate.
  * Every change is idempotent: its verify_after predicates are checked before and after, so
    re-running the engine on an already-ported tree is a no-op, and the same engine is used
    for a fresh apply and for a re-apply after an upstream update.
  * The mechanical NINFER_SM86 compatibility sweep derives its file set from the live tree
    (`git ls-files`); no file list or line number is stored in the manifest.

Usage:
    python3 tools/port_engine.py [--worktree PATH] [--manifest PATH]
                                 [--no-commit] [--report PATH] [--quiet]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    REPO_ROOT,
    die,
    git,
    load_manifest,
    log_fail,
    log_ok,
    log_step,
    log_warn,
    parse_index_line,
    patch_target_relpath,
    read_file_lines,
    read_text,
    write_text,
)


# ---------------------------------------------------------------------------
# predicates
# ---------------------------------------------------------------------------

def _iter_tree(worktree: Path, paths: list[str]):
    """Yield tracked relative file paths under the given top-level directories."""
    for prefix in paths or ["."]:
        proc = git(worktree, "ls-files", "--", prefix, check=False)
        for line in (proc.stdout or "").splitlines():
            line = line.strip()
            if line:
                candidate = worktree / line
                if candidate.is_file():
                    yield line


def _safe_text(path: Path) -> str:
    try:
        text, _ = read_file_lines(path)
        return text
    except Exception:
        return ""


def run_verify(worktree: Path, verify: dict | None, relpath: str | None = None) -> tuple[bool, list[str]]:
    """Evaluate a change's verify_after block. Returns (ok, problems)."""
    if not verify:
        return True, []
    problems: list[str] = []

    def scoped_file():
        """(text, problem) for the change's own file. An undecodable file is a problem, not a
        vacuous pass - a binary or mis-encoded file must not satisfy must_not_match checks."""
        candidate = worktree / relpath
        if not candidate.is_file():
            return None, f"file missing: {relpath}"
        try:
            text, _ = read_file_lines(candidate)
            return text, None
        except UnicodeDecodeError:
            return None, f"file not valid UTF-8: {relpath}"

    for pattern in verify.get("must_match", []):
        if not relpath:
            continue
        text, problem = scoped_file()
        if problem:
            problems.append(problem)
            continue
        if text is not None and not re.search(pattern, text, re.MULTILINE):
            problems.append(f"must_match not found: {pattern!r} in {relpath}")

    for pattern in verify.get("must_not_match", []):
        if not relpath:
            continue
        text, problem = scoped_file()
        if problem:
            problems.append(problem)
            continue
        if text is not None and re.search(pattern, text, re.MULTILINE):
            problems.append(f"must_not_match still present: {pattern!r} in {relpath}")

    for pattern in verify.get("tree_must_not_match", []):
        for rel in _iter_tree(worktree, verify.get("paths", change_paths_from_verify(verify))):
            if re.search(pattern, _safe_text(worktree / rel), re.MULTILINE):
                problems.append(f"tree must_not_match still present: {pattern!r} in {rel}")
                break

    return not problems, problems


def change_paths_from_verify(verify: dict) -> list[str]:
    return verify.get("paths", ["."])


# ---------------------------------------------------------------------------
# apply strategies
# ---------------------------------------------------------------------------

def strategy_patch_2way(worktree: Path, change: dict, patch_text: str) -> tuple[bool, str]:
    patch_file = worktree / ".port-tmp-patch.txt"
    write_text(patch_file, patch_text, crlf=False)
    try:
        check = git(worktree, "apply", "--check", "--whitespace=nowarn", str(patch_file), check=False)
        if check.returncode != 0:
            return False, (check.stderr or check.stdout).strip()
        apply = git(worktree, "apply", "--whitespace=nowarn", str(patch_file), check=False)
        if apply.returncode != 0:
            return False, (apply.stderr or apply.stdout).strip()
        return True, "direct 2-way git apply"
    finally:
        patch_file.unlink(missing_ok=True)


def _subprocess(cwd: Path, cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace"
    )


def strategy_patch_3way(worktree: Path, change: dict, patch_text: str) -> tuple[bool, str]:
    """Explicit three-way merge.

    base    = the blob the patch was generated from (from its `index` line),
    current = the worktree file,
    theirs  = base + patch (reconstructed in a scratch directory).

    This uses `git merge-file` rather than `git apply --3way` because its semantics are
    transparent: exit 0 is a clean merge, any other value is the number of conflicts. It does
    not depend on git apply's index-matching quirks, and the failure is a count a human can
    reason about.
    """
    base_sha, _result_sha = parse_index_line(patch_text)
    relpath = patch_target_relpath(patch_text)
    if not base_sha or not relpath:
        return False, "patch carries no resolvable index line"
    base_proc = git(worktree, "cat-file", "blob", base_sha, check=False)
    if base_proc.returncode != 0:
        return False, f"base blob {base_sha[:12]} not in object store (clone lacks the base commit)"

    target = worktree / relpath
    if not target.is_file():
        return False, f"file missing in worktree: {relpath}"
    try:
        current, crlf = read_file_lines(target)
    except UnicodeDecodeError:
        return False, f"file not valid UTF-8: {relpath}"
    base_text = base_proc.stdout
    if "\r\n" in base_text:
        base_text = base_text.replace("\r\n", "\n")

    with tempfile.TemporaryDirectory(prefix="ninfer-l20-3way-") as tmp:
        tmp_path = Path(tmp)
        tmp_file = tmp_path / relpath
        tmp_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file.write_text(base_text, encoding="utf-8", newline="")
        patch_copy = tmp_path / "change.patch"
        patch_copy.write_text(patch_text, encoding="utf-8", newline="")
        applied = _subprocess(tmp_path, ["git", "apply", "--whitespace=nowarn", "change.patch"])
        if applied.returncode != 0:
            return False, "could not reconstruct patched base in scratch dir: " + (applied.stderr or "").strip()
        theirs_text = tmp_file.read_text(encoding="utf-8")

        ours_file = tmp_path / "ours"
        base_file = tmp_path / "base"
        theirs_file = tmp_path / "theirs"
        ours_file.write_text(current, encoding="utf-8", newline="")
        base_file.write_text(base_text, encoding="utf-8", newline="")
        theirs_file.write_text(theirs_text, encoding="utf-8", newline="")
        merged = _subprocess(
            tmp_path, ["git", "merge-file", "-p",
                       str(ours_file), str(base_file), str(theirs_file)]
        )
        # exit 0 = clean merge on stdout; 1..n = n conflicts (with -p, a failed merge still
        # prints the conflicted text, so the marker check below is the authoritative test);
        # 128/129 = usage error, which must not be mistaken for "129 conflicts".
        if merged.returncode >= 128:
            return False, f"git merge-file usage error ({merged.returncode}): " + (merged.stderr or "").strip()
        if merged.returncode != 0:
            return False, f"three-way merge has {merged.returncode} conflict(s)"
        if "<<<<<<<" in merged.stdout or ">>>>>>>" in merged.stdout:
            return False, "three-way merge produced conflict markers"

    write_text(target, merged.stdout, crlf=crlf)
    return True, "three-way merge (base from patch index line)"


def strategy_transforms(worktree: Path, change: dict) -> tuple[bool, str]:
    """Apply the first variant whose steps all match with their expected counts.

    A variant is a list of find/count/replace steps; a step that misses (0 matches, or more
    than expected) means this variant does not describe the current source shape, so the next
    variant is tried. Steps of a committed variant run in order on the same in-memory text, so
    multi-step shapes (a constant and all of its consumers) are rewritten atomically.
    """
    relpath = change["file"]
    target = worktree / relpath
    if not target.is_file():
        return False, f"file missing in worktree: {relpath}"
    try:
        text, crlf = read_file_lines(target)
    except UnicodeDecodeError:
        return False, f"file not valid UTF-8: {relpath}"

    for variant in change.get("transforms", []):
        steps = variant.get("steps")
        if steps is None:
            steps = [variant]
        candidate = text
        ok = True
        for step in steps:
            pattern = step["find"]
            expected = int(step.get("count", 1))
            matches = re.findall(pattern, candidate)
            if len(matches) != expected:
                ok = False
                break
            replacement = step["replace"]
            # Direct template replacement: \n in the manifest text becomes a newline, \1..\9
            # become group references - exactly what the manifest authors write.
            candidate = re.sub(pattern, replacement, candidate, count=expected)
        if not ok:
            continue
        write_text(target, candidate, crlf=crlf)
        label = variant.get("id") or "transform"
        return True, f"transform variant: {label}"

    return False, "no transform variant matched the current source shape"


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def apply_sweep(worktree: Path, change: dict) -> tuple[bool, str, int]:
    """Rewrite bare NINFER_SM86 directives into the SM86||SM89 compatibility form, file by file,
    with the file set derived from the live tree. Idempotent: a line that already carries
    NINFER_SM89 is left alone, so re-runs never append a duplicate clause."""
    touched: list[str] = []
    rules = change.get("rules", [])
    for rel in _iter_tree(worktree, change.get("paths", ["."])):
        target = worktree / rel
        try:
            text, crlf = read_file_lines(target)
        except UnicodeDecodeError:
            continue  # binary fixture - nothing to sweep
        out_lines = []
        changed = False
        for line in text.split("\n"):
            stripped = line.strip()
            new_line = line
            for old, new in rules:
                if stripped == old and "NINFER_SM89" not in line:
                    indent = line[: len(line) - len(line.lstrip())]
                    new_line = indent + new
                    changed = True
                    break
            out_lines.append(new_line)
        if changed:
            write_text(target, "\n".join(out_lines), crlf=crlf)
            touched.append(rel)
    message = f"sweep rewrote {len(touched)} file(s)"
    if touched:
        message += ": " + ", ".join(touched)
    return True, message, len(touched)


# ---------------------------------------------------------------------------
# drift report
# ---------------------------------------------------------------------------

def drift_diagnostic(worktree: Path, change: dict) -> dict:
    """Collect context for a change that could not be applied: the expected anchor and what the
    file actually contains around the nearest match of its prefix. This is what a human (or a
    future patch regeneration) needs to extend the variant ladder or regenerate the patch."""
    relpath = change.get("file")
    info: dict = {"file": relpath, "hint": "see docs/UPGRADING.md"}
    if not relpath or not (worktree / relpath).is_file():
        info["state"] = "file-missing"
        return info
    text = _safe_text(worktree / relpath)
    variants = change.get("transforms", [])
    if not variants:
        info["state"] = "no-anchor"
        return info
    first_step = (variants[0].get("steps") or [variants[0]])[0]
    pattern = first_step.get("find", "")
    # Strip regex metacharacters to a plain-text prefix we can probe the file with.
    prefix = re.sub(r"\\.", "x", pattern)
    prefix = re.sub(r"[(){}\[\]^$*+?|]", "x", prefix)[:60]
    probe = re.escape(prefix[:40])
    match = re.search(probe, text)
    info["state"] = "context-drifted" if match else "anchor-gone"
    if match:
        start = max(0, match.start() - 400)
        end = min(len(text), match.end() + 600)
        info["current_context"] = text[start:end]
    info["expected_prefix"] = prefix
    return info


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", default=str(REPO_ROOT / "src"),
                        help="upstream checkout to port (default: <repo>/src)")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--no-commit", action="store_true",
                        help="leave the port uncommitted in the worktree")
    parser.add_argument("--report", default=None,
                        help="where to write the JSON report (default: <worktree>/.port-report.json)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    worktree = Path(args.worktree).resolve()
    if not (worktree / "CMakeLists.txt").exists():
        die(f"worktree not found: {worktree} (clone the upstream tree there first, see scripts/build.sh)")
    manifest = load_manifest(Path(args.manifest) if args.manifest else None)
    report_path = Path(args.report) if args.report else worktree / ".port-report.json"

    base_sha = git(worktree, "rev-parse", "HEAD").stdout.strip()
    base_ref = git(worktree, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    log_step(f"port {manifest['port_id']} v{manifest['port_version']}")
    log_ok(f"worktree {worktree}")
    log_ok(f"upstream base: {base_sha[:12]} ({base_ref})")

    results = []
    for change in manifest["changes"]:
        change_id = change["id"]
        log_step(f"{change_id}")

        relpath = change.get("file")
        verify = change.get("verify_after")
        ok_before, _ = run_verify(worktree, verify, relpath)
        if ok_before:
            log_ok("already applied (verify_after satisfied) - skipping")
            results.append({"id": change_id, "status": "present", "strategy": "none"})
            continue

        ok = False
        strategy = None
        if change["kind"] == "patch":
            patch_file = REPO_ROOT / change["patch"]
            patch_text = read_text(patch_file) if patch_file.is_file() else None
            ladder = []
            if patch_text is not None:
                ladder.append(("2way", lambda: strategy_patch_2way(worktree, change, patch_text)))
                ladder.append(("3way", lambda: strategy_patch_3way(worktree, change, patch_text)))
            else:
                if not args.quiet:
                    log_warn(f"no patch file yet ({change['patch']}) - relying on transforms; "
                             "run tools/regen_patches.py afterwards to record it")
            ladder.append(("transform", lambda: strategy_transforms(worktree, change)))
            for name, func in ladder:
                ok, message = func()
                if ok:
                    strategy = message
                    break
                if not args.quiet:
                    log_warn(f"{name}: {message.splitlines()[0] if message else 'no details'}")
            if not ok:
                results.append({
                    "id": change_id, "status": "drift",
                    "diagnostic": drift_diagnostic(worktree, change),
                    "hint": "upstream changed this file beyond every known shape - see docs/UPGRADING.md",
                })
                log_fail(f"DRIFT: {message}")
                continue
            log_ok(strategy)
        elif change["kind"] == "sweep":
            ok, strategy, _ = apply_sweep(worktree, change)
            log_ok(strategy)
        else:
            die(f"unknown change kind: {change['kind']}")

        ok_after, problems = run_verify(worktree, verify, relpath)
        status = "applied" if ok_after else "applied-unverified"
        entry = {"id": change_id, "status": status, "strategy": strategy}
        if problems:
            entry["verify_problems"] = problems
        results.append(entry)
        if ok_after:
            log_ok("verify_after satisfied")
        else:
            log_fail(f"verify_after problems: {problems}")

    committed = False
    if not args.no_commit:
        # keep the report file itself out of the port commit
        exclude_file = worktree / ".git" / "info" / "exclude"
        exclude_dir = exclude_file.parent
        exclude_dir.mkdir(parents=True, exist_ok=True)
        existing = exclude_file.read_text(encoding="utf-8") if exclude_file.exists() else ""
        if ".port-report.json" not in existing and ".port-tmp-patch.txt" not in existing:
            with open(exclude_file, "a", encoding="utf-8") as handle:
                handle.write("\n# ninfer-l20 port engine artifacts\n.port-report.json\n.port-tmp-patch.txt\n")
        status_porcelain = git(worktree, "status", "--porcelain").stdout.strip()
        if status_porcelain:
            git(worktree, "add", "-A")
            # The port commit is machine-generated: honour the user's identity, fall back to a
            # dedicated one so the commit never fails on a fresh machine.
            fallback_config = []
            for key, fallback in (("user.name", f"{manifest['port_id']} port engine"),
                                  ("user.email", f"port@{manifest['port_id']}.local")):
                if not git(worktree, "config", key, check=False).stdout.strip():
                    fallback_config.append((key, fallback))
            git(worktree, "commit", "-m",
                f"{manifest['layout']['port_commit_prefix']}: apply port v{manifest['port_version']} "
                f"on upstream {base_sha[:12]}",
                config=fallback_config)
            committed = True
            log_ok("port committed on top of upstream")
        else:
            log_ok("nothing to commit (port already in place)")

    report = {
        "port_id": manifest["port_id"],
        "port_version": manifest["port_version"],
        "upstream_base": base_sha,
        "upstream_ref": base_ref,
        "committed": committed,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "changes": results,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    failed = [r for r in results if r["status"] not in ("applied", "present")]
    log_step("port summary")
    for entry in results:
        marker = {"applied": " ok ", "present": "skip", "drift": "DRIFT",
                  "applied-unverified": "WARN"}.get(entry["status"], "?")
        print(f"  [{marker}] {entry['id']}: {entry['status']}"
              + (f" ({entry['strategy']})" if entry.get("strategy") else ""))
    log_ok(f"report: {report_path}")
    if failed:
        log_fail(f"{len(failed)} change(s) need attention - details in {report_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
