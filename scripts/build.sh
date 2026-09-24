#!/usr/bin/env bash
# ============================================================
# ninfer-l20 : fetch the upstream tree, apply the port, build.
#
# Ubuntu-first (the original port required WSL2 + PowerShell; this
# runs on bare Ubuntu 24.04, in WSL, or inside docker/Dockerfile).
#
# Nothing here is pinned:
#   * the upstream branch tip is the default base (no commit SHA input),
#   * the CUDA arch defaults to the manifest value but is checked
#     against the GPU actually present (nvidia-smi),
#   * the CUDA toolkit and host compiler are auto-located,
#   * provenance (upstream SHA, port commit, toolchain, GPU) is
#     recorded in build-info.json after a successful build.
#
# Usage:
#   bash scripts/setup.sh        # once, on a fresh machine
#   bash scripts/build.sh        # clone + port + configure + build
#   bash scripts/build.sh --test            # + ctest smoke run
#   bash scripts/build.sh --configure-only  # stop after cmake configure
#   bash scripts/build.sh --fresh           # re-clone the upstream tree
#
# Environment overrides (all optional):
#   NINFER_L20_ROOT   project root (default: this checkout)
#   NINFER_CUDA_ARCH  CMAKE_CUDA_ARCHITECTURES (default: manifest)
#   CUDA_HOME         CUDA toolkit root (default: auto-locate)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NINFER_L20_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MANIFEST="$ROOT/port/manifest.json"
SRC="$ROOT/src"
PYTHON="${PYTHON:-python3}"

FRESH=0; TEST=0; CONFIGURE_ONLY=0
for a in "$@"; do
  case "$a" in
    --fresh) FRESH=1 ;;
    --test) TEST=1 ;;
    --configure-only) CONFIGURE_ONLY=1 ;;
    *) echo "unknown arg: $a (see header)"; exit 2 ;;
  esac
done

step() { printf '\n########## %s ##########\n' "$*"; }
die()  { printf 'FATAL: %s\n' "$*" >&2; exit 1; }

[ -f "$MANIFEST" ] || die "manifest not found: $MANIFEST"

# --- manifest values -------------------------------------------------
eval "$("$PYTHON" - "$MANIFEST" <<'EOF'
import json, shlex, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
base, lay, build = m["base"], m["layout"], m["build"]
floors = m["require"]["pkg_config_floors"]
print("UPSTREAM_REPO=" + shlex.quote(base["repo"]))
print("UPSTREAM_BRANCH=" + shlex.quote(base["branch"]))
print("BUILD_SUBDIR=" + shlex.quote(lay["build_subdir"]))
print("PORT_BRANCH=" + shlex.quote(lay["port_branch"]))
print("CUDA_ARCH_DEFAULT=" + shlex.quote(str(build["cuda_arch_default"])))
print("CMAKE_MIN=" + shlex.quote(m["require"]["cmake"]))
print("CMAKE_OPTIONS=" + shlex.quote(" ".join(build["cmake_options"])))
print("PC_FLOORS=" + shlex.quote(" ".join(f"{k}:{v}" for k, v in floors.items())))
EOF
)"
BUILD_DIR="$SRC/$BUILD_SUBDIR"

step "preflight"
for t in git cmake ninja "$PYTHON" pkg-config; do
  command -v "$t" >/dev/null 2>&1 || die "$t missing (run: bash scripts/setup.sh)"
done

# CUDA toolkit: honour CUDA_HOME, else PATH, else /usr/local/cuda*.
if [ -z "${CUDA_HOME:-}" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  else
    for d in /usr/local/cuda-*/; do
      [ -x "${d}bin/nvcc" ] && { CUDA_HOME="${d%/}"; break; }
    done
  fi
fi
[ -x "$CUDA_HOME/bin/nvcc" ] || die "no CUDA toolkit found (set CUDA_HOME)"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
NVCC_VER="$("$CUDACXX" --version | tail -1 | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
printf '  nvcc   CUDA %s (%s)\n' "$NVCC_VER" "$CUDA_HOME"
printf '  cmake  %s\n' "$(cmake --version | head -1 | sed 's/cmake //')"

# host compiler: gcc-13 if present (Ubuntu 24.04 default), else the distro default.
if command -v gcc-13 >/dev/null 2>&1; then
  HOST_CC=gcc-13; HOST_CXX=g++-13
else
  HOST_CC="$(command -v cc || echo gcc)"; HOST_CXX="$(command -v c++ || echo g++)"
  printf '  [warn] gcc-13 not found; using %s (%s)\n' "$HOST_CC" "$($HOST_CXX -dumpversion 2>/dev/null || echo '?')"
fi
printf '  host   %s %s\n' "$HOST_CXX" "$($HOST_CXX -dumpversion 2>/dev/null || echo '?')"

# pkg-config floors (manifest-driven) - fail early with a clear message.
fail=0
IFS=':' read -ra FLOOR_PAIRS <<< "$PC_FLOORS"
for pair in "${FLOOR_PAIRS[@]}"; do
  mod="${pair%%:*}"; floor="${pair##*:}"
  if pkg-config --exists "$mod" 2>/dev/null && pkg-config --atleast-version="$floor" "$mod" 2>/dev/null; then
    printf '  %-12s OK   %s\n' "$mod" "$(pkg-config --modversion "$mod")"
  else
    printf '  %-12s FAIL %s (need >=%s)\n' "$mod" "$(pkg-config --modversion "$mod" 2>/dev/null || echo missing)" "$floor"
    fail=1
  fi
done
[ "$fail" -eq 0 ] || die "pkg-config floors not met (see above) - on 22.04 this means: use Ubuntu 24.04"

# --- GPU + architecture ----------------------------------------------
ARCH="${NINFER_CUDA_ARCH:-$CUDA_ARCH_DEFAULT}"
step "target device"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LINE="$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | head -1)"
  printf '  GPU: %s\n' "$GPU_LINE"
  GPU_CC="$(printf '%s' "$GPU_LINE" | cut -d, -f2 | tr -d ' ')"
  GPU_ARCH="${GPU_CC//./}"
  if [ "$GPU_ARCH" != "$ARCH" ]; then
    printf '  [warn] build targets sm_%s but this GPU is sm_%s - the build still targets the manifest arch; ' \
      "the runtime device_sm_count() reads this GPU's real SM count\n" "$ARCH" "$GPU_ARCH"
  fi
else
  printf '  [info] no nvidia-smi - building for the manifest arch (sm_%s) without a device check\n' "$ARCH"
fi
printf '  arch: CMAKE_CUDA_ARCHITECTURES=%s\n' "$ARCH"

# --- source tree ------------------------------------------------------
step "source tree"
if [ "$FRESH" -eq 1 ] && [ -d "$SRC" ]; then rm -rf "$SRC"; fi
if [ ! -d "$SRC/.git" ]; then
  # Full history (not --depth 1): the port engine's three-way fallback needs the base blobs
  # of the recorded patches in the object store. The repo is small.
  rm -rf "$SRC"
  git clone --quiet --branch "$UPSTREAM_BRANCH" "$UPSTREAM_REPO" "$SRC"
  printf '  cloned %s @ %s\n' "$UPSTREAM_REPO" "$UPSTREAM_BRANCH"
else
  printf '  reusing %s @ %s\n' "$SRC" "$(git -C "$SRC" rev-parse --short HEAD)"
fi
UPSTREAM_SHA="$(git -C "$SRC" rev-parse HEAD)"
UPSTREAM_REF="$(git -C "$SRC" rev-parse --abbrev-ref HEAD)"
[ "$UPSTREAM_REF" = "HEAD" ] && UPSTREAM_REF="(detached at ${UPSTREAM_SHA:0:12})"

# --- port -------------------------------------------------------------
step "apply the L20 port"
"$PYTHON" "$ROOT/tools/port_engine.py" --worktree "$SRC"
PORT_REPORT="$SRC/.port-report.json"
[ -f "$PORT_REPORT" ] || die "port report missing (the engine did not finish)"

step "verify port invariants"
"$PYTHON" "$ROOT/tools/verify.py" --worktree "$SRC"

# --- configure + build -------------------------------------------------
step "configure"
cmake -S "$SRC" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCMAKE_C_COMPILER="$HOST_CC" \
  -DCMAKE_CXX_COMPILER="$HOST_CXX" \
  -DCMAKE_CUDA_HOST_COMPILER="$HOST_CXX" \
  $CMAKE_OPTIONS

if [ "$CONFIGURE_ONLY" -eq 1 ]; then
  step "configure-only: stopping before build"
  exit 0
fi

step "build"
# No numeric -j: the upstream convention is unrestricted build parallelism.
cmake --build "$BUILD_DIR" -j

if [ "$TEST" -eq 1 ]; then
  step "test (smoke)"
  ctest --test-dir "$BUILD_DIR" --output-on-failure || {
    echo "FATAL: ctest reported failures"
    exit 1
  }
fi

# --- provenance --------------------------------------------------------
step "binaries"
find "$BUILD_DIR" -maxdepth 3 -type f -executable \
  \( -name 'ninfer' -o -name 'ninfer-serve' -o -name 'ninfer_bench' \) \
  -printf '  %p\n' 2>/dev/null | sort

PORT_COMMIT="$(git -C "$SRC" log --format=%H -1 --grep="l20-port" || true)"
GPU_JSON="null"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_JSON="$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | head -1 | \
    "$PYTHON" -c "
import sys
name, cc, mem = [x.strip() for x in sys.stdin.read().split(',')]
print('{\"name\": %r, \"compute_cap\": %r, \"memory_total\": %r}' % (name, cc, mem))
  " 2>/dev/null || echo null)"
fi

step "provenance"
"$PYTHON" - "$ROOT/build-info.json" <<EOF
import json, datetime, sys, os
info = {
  "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
  "upstream": {"repo": "$UPSTREAM_REPO", "ref": "$UPSTREAM_REF", "sha": "$UPSTREAM_SHA"},
  "port_commit": "$PORT_COMMIT" or None,
  "cuda_arch": int("$ARCH"),
  "toolchain": {"nvcc": "$NVCC_VER", "cuda_home": "$CUDA_HOME", "host_compiler": "$HOST_CXX"},
  "gpu": $GPU_JSON,
  "port_report": "$PORT_REPORT",
  "build_dir": "$BUILD_DIR",
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)
print(json.dumps(info, indent=2))
EOF

step "BUILD COMPLETE"
echo "  serve:    bash scripts/start.sh [port]"
echo "  upgrade:  bash scripts/upgrade.sh   (when the upstream moves)"
