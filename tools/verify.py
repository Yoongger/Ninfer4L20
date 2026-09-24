#!/usr/bin/env python3
"""Static verification of the ninfer-l20 port.

This is the layer that makes the port's invariants enforceable after an upstream update:
every check is a property of the tree, not of a commit or a patch. If upstream changes
anything that the port relies on, exactly one of these checks goes red and tells you which
file regressed - instead of discovering it inside a 1400-file CUDA build.

Checks (all derived from the port spec, docs/PORT-SPEC.md):
  V1  architecture identity: top-level CMakeLists.txt defines NINFER_SM86 / NINFER_SM89 at
      directory scope from CMAKE_CUDA_ARCHITECTURES.
  V2  no hard-coded identity: src/CMakeLists.txt no longer force-defines NINFER_SM86.
  V3  no bare NINFER_SM86 preprocessor conditional anywhere in the tree (every compat switch
      must accept NINFER_SM89 as well) - the invariant the dynamic sweep maintains.
  V4  no per-part SM-count literal in src/ (the class of constant that made the L20 port
      necessary: kRtx5090SmCount = 170 and friends).
  V5  device_sm_count() is declared in src/core/device.h and defined in src/core/device.cu.
  V6  the GDN chunked output kernel sizes its wave from device_sm_count().
  V7  the sparse-MoE prefill cap is runtime-derived.
  V8  the cooperative residency constant for the 27B GDN gating projection is split-aware.

Usage:
    python3 tools/verify.py [--worktree PATH] [--quiet]
Exit status: 0 = all invariants hold, 1 = at least one violation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import die, git, log_fail, log_ok, log_step, log_warn  # noqa: E402
from common import REPO_ROOT  # noqa: E402
from port_engine import _iter_tree, _safe_text  # noqa: E402

RESULTS: list[tuple[str, str, bool, str]] = []


def record(check: str, title: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((check, title, ok, detail))
    marker = " ok " if ok else "FAIL"
    print(f"  [{marker}] {check} {title}" + (f" - {detail}" if detail and not ok else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", default=str(REPO_ROOT / "src"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    worktree = Path(args.worktree).resolve()
    if not (worktree / "CMakeLists.txt").exists():
        die(f"worktree not found: {worktree}")

    base_sha = git(worktree, "rev-parse", "HEAD").stdout.strip()
    log_step(f"verify port invariants @ {worktree} (upstream {base_sha[:12]})")

    top_cmake = _safe_text(worktree / "CMakeLists.txt")
    src_cmake = _safe_text(worktree / "src" / "CMakeLists.txt")

    # V1
    ok = bool(
        re.search(r"if\(CMAKE_CUDA_ARCHITECTURES STREQUAL \"86\"\)\s*\n\s*add_compile_definitions\(NINFER_SM86=1\)", top_cmake)
        and re.search(r"elseif\(CMAKE_CUDA_ARCHITECTURES STREQUAL \"89\"\)\s*\n\s*add_compile_definitions\(NINFER_SM89=1\)", top_cmake)
    )
    record("V1", "architecture identity defined at directory scope", ok,
           "top-level CMakeLists.txt must map CMAKE_CUDA_ARCHITECTURES 86/89 to NINFER_SM86/NINFER_SM89")

    # V2
    ok = "target_compile_definitions(ninfer_ops PUBLIC NINFER_SM86=1)" not in src_cmake
    record("V2", "no hard-coded NINFER_SM86 in src/CMakeLists.txt", ok)

    # V3 - every NINFER_SM86 preprocessor conditional also accepts NINFER_SM89
    bare = re.compile(r"^\s*#\s*(if\s+defined\(NINFER_SM86\)\s*$|if\s+!defined\(NINFER_SM86\)\s*$|ifdef NINFER_SM86\s*$|ifndef NINFER_SM86\s*$)")
    offenders = []
    for rel in _iter_tree(worktree, ["src", "tests", "apps", "include", "tools", "bench"]):
        text = _safe_text(worktree / rel)
        for lineno, line in enumerate(text.split("\n"), start=1):
            if bare.match(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    record("V3", "no bare NINFER_SM86 preprocessor conditional", not offenders,
           "; ".join(offenders[:5]) + (f" (+{len(offenders) - 5} more)" if len(offenders) > 5 else ""))

    # V4 - no per-part SM-count literal in src/
    literal = re.compile(r"\bSmCount\b\s*=\s*\d+|sm_count\s*=\s*(82|92|128|142|170)\b")
    offenders = []
    for rel in _iter_tree(worktree, ["src"]):
        text = _safe_text(worktree / rel)
        for lineno, line in enumerate(text.split("\n"), start=1):
            if re.search(r"\bSmCount\b", line) and literal.search(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    record("V4", "no per-part SM-count literal in src/", not offenders,
           "; ".join(offenders[:5]))

    # V5
    device_h = _safe_text(worktree / "src" / "core" / "device.h")
    device_cu = _safe_text(worktree / "src" / "core" / "device.cu")
    ok = re.search(r"^int device_sm_count\(\);\s*$", device_h, re.MULTILINE) is not None \
        and re.search(r"^int device_sm_count\(\) \{", device_cu, re.MULTILINE) is not None \
        and "props.multiProcessorCount" in device_cu
    record("V5", "device_sm_count() declared and defined", ok,
           "src/core/device.h must declare, src/core/device.cu must define, the cached runtime query")

    # V6
    output_cu = _safe_text(worktree / "src" / "ops" / "linear_attention" / "gated_delta_net" / "chunked" / "output.cu")
    ok = "kCtasPerSm * device_sm_count()" in output_cu and "kRtx5090SmCount" not in output_cu and "kTargetCtas" not in output_cu
    record("V6", "GDN chunked output wave is runtime-sized", ok)

    # V7
    sparse = _safe_text(worktree / "src" / "ops" / "sparse_moe" / "prefill" / "sparse_moe_prefill_kernels.cu")
    ok = "kRtx5090SmCount" not in sparse and re.search(r"prefill_(max|persistent)_blocks\(\)", sparse) is not None
    record("V7", "sparse-MoE prefill grid cap is runtime-derived", ok)

    # V8
    gating = _safe_text(worktree / "src" / "ops" / "gdn_gating_proj" / "bf16" / "bf16_gdn_gating_proj_kernels.cu")
    ok = re.search(r"return SplitK == 8 \? 2 : 1;", gating) is not None
    record("V8", "cooperative residency is split-aware for Bf16Gdn27Geometry", ok,
           "bf16_gdn_gating_proj_kernels.cu must return SplitK == 8 ? 2 : 1 (the L20 cooperative-launch crash fix)")

    failed = [r for r in RESULTS if not r[2]]
    log_step("verify summary")
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} invariants hold")
    if failed:
        for check, title, _ok, detail in failed:
            log_fail(f"{check} {title}" + (f" - {detail}" if detail else ""))
        return 1
    log_ok("all port invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
