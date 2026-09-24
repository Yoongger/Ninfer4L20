#!/usr/bin/env bash
# ============================================================
# ninfer-l20 : install build prerequisites on Ubuntu and gate
# on the version floors the upstream project declares.
#
# Ubuntu-first: no WSL, no /mnt/d, no PowerShell in the path.
# The version floors come from port/manifest.json (single source
# of truth) - nothing in this script hard-codes a version.
#
# Usage:
#   bash scripts/setup.sh            # install + verify
#   bash scripts/setup.sh --check    # verify only (no apt)
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$ROOT/port/manifest.json"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

log() { printf '\n=== %s ===\n' "$*"; }

[ -f "$MANIFEST" ] || { echo "FATAL: manifest not found: $MANIFEST"; exit 1; }

# --- manifest values (python3 is a hard dependency of the port engine anyway) ---
command -v python3 >/dev/null 2>&1 || { echo "FATAL: python3 is required (the port engine is Python 3 stdlib)"; exit 1; }
python3 -c "import json; json.load(open('$MANIFEST', encoding='utf-8'))" \
  || { echo "FATAL: manifest is not valid JSON: $MANIFEST"; exit 1; }
REQ_CMAKE="$(python3 -c "import json; print(json.load(open('$MANIFEST', encoding='utf-8'))['require']['cmake'])")"

# --- OS ---
log "distribution"
if [ -r /etc/os-release ]; then
  . /etc/os-release
  echo "  ID=$ID VERSION=$VERSION_ID"
  if [ "$ID" = "ubuntu" ]; then
    case "$VERSION_ID" in
      24.*) echo "  Ubuntu 24.04 series: satisfies all version floors from its own repositories." ;;
      22.*) echo "  [warn] Ubuntu 22.04: FFmpeg/libcurl floors will fail the gate below (FFmpeg 4.4, libcurl 7.81)." ;;
      *)    echo "  [warn] Ubuntu $VERSION_ID: not tested; the gate below is authoritative." ;;
    esac
  else
    echo "  [warn] not Ubuntu ($ID). The build is Ubuntu-based; other distros may work if the floors pass."
  fi
else
  echo "  [warn] /etc/os-release not found (running inside a container?); the floors below are authoritative."
fi

if [ "$CHECK_ONLY" -eq 0 ]; then
  export DEBIAN_FRONTEND=noninteractive
  log "apt update"
  apt-get update -qq || { echo "apt-get update FAILED"; exit 1; }

  log "base toolchain"
  apt-get install -y -qq --no-install-recommends \
    build-essential ninja-build pkg-config git curl ca-certificates python3 \
    || { echo "base install FAILED"; exit 1; }

  log "media + network dev libraries"
  # The top-level CMakeLists takes the pkg-config route on Linux and requires these.
  apt-get install -y -qq --no-install-recommends \
    libcurl4-openssl-dev \
    libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libswresample-dev \
    libavfilter-dev libavdevice-dev \
    || echo "WARNING: media dev install reported an error; the floors below will say whether it is fatal"

  log "GCC 13 (host compiler)"
  # Ubuntu 24.04 ships gcc-13 as the default toolchain; on older pockets add it explicitly.
  if ! command -v gcc-13 >/dev/null 2>&1; then
    apt-get install -y -qq gcc-13 g++-13 || echo "WARNING: gcc-13 not installed; the gate below will say if it matters"
  fi
fi

# --- tool presence + version floors ---
fail=0
log "verification"
need_cmd() {
  local name="$1" got
  if command -v "$name" >/dev/null 2>&1; then
    printf '  %-14s OK   %s\n' "$name" "$("$name" --version 2>&1 | head -1)"
  else
    printf '  %-14s MISSING\n' "$name"; fail=1
  fi
}
need_cmd git
need_cmd cmake
need_cmd ninja
need_cmd pkg-config
need_cmd python3
command -v gcc-13 >/dev/null 2>&1 \
  && printf '  %-14s OK   %s\n' "gcc-13" "$(gcc-13 -dumpversion)" \
  || { printf '  %-14s MISSING\n' "gcc-13"; fail=1; }

# CUDA toolkit: honour CUDA_HOME, else PATH, else the NEWEST /usr/local/cuda* with an nvcc
# (version-sorted, so the `cuda` / `cuda-13` convenience symlinks are not picked by accident).
cuda_home="${CUDA_HOME:-}"
if [ -z "$cuda_home" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    cuda_home="$(dirname "$(dirname "$(command -v nvcc)")")"
  else
    best=""
    for d in /usr/local/cuda-*/; do
      d="${d%/}"
      [ -x "$d/bin/nvcc" ] || continue
      [ -z "$best" ] && best="$d" && continue
      [ "$(printf '%s\n%s\n' "$best" "$d" | sort -V | tail -1)" = "$d" ] && best="$d"
    done
    [ -n "$best" ] && cuda_home="$best"
  fi
fi
if [ -n "$cuda_home" ] && [ -x "$cuda_home/bin/nvcc" ]; then
  nvcc_ver="$("$cuda_home/bin/nvcc" --version | grep -oE 'release [0-9][0-9.]*' | head -1 | cut -d' ' -f2)"
  printf '  %-14s OK   CUDA %s at %s\n' "nvcc" "${nvcc_ver:-?}" "$cuda_home"
else
  printf '  %-14s MISSING (no nvcc on PATH or under /usr/local/cuda*)\n' "nvcc"; fail=1
fi

# pkg-config floors, read from the manifest.
need_pc() {
  local mod="$1" min="$2" got
  if ! pkg-config --exists "$mod" 2>/dev/null; then
    printf '  %-14s MISSING (needs >=%s)\n' "$mod" "$min"; fail=1; return
  fi
  got="$(pkg-config --modversion "$mod")"
  if pkg-config --atleast-version="$min" "$mod" 2>/dev/null; then
    printf '  %-14s OK   %s\n' "$mod" "$got"
  else
    printf '  %-14s TOO OLD  %s (needs >=%s)\n' "$mod" "$got" "$min"; fail=1
  fi
}
while read -r tag mod floor; do
  [ "$tag" = "REQ_PC" ] && need_pc "$mod" "$floor"
done < <(python3 -c "
import json
m = json.load(open('$MANIFEST', encoding='utf-8'))
for k, v in m['require']['pkg_config_floors'].items():
    print('REQ_PC', k, v)")

# cmake floor (tolerates both "cmake 3.28" and the newer "cmake version 3.28" output)
cmake_ver="$(cmake --version 2>/dev/null | head -1 | grep -oE '[0-9]+(\.[0-9]+)+' | head -1)"
if [ -n "$cmake_ver" ] && python3 -c "
import sys
a = [int(x) for x in '$cmake_ver'.split('.')[:2] if x.isdigit()]
b = [int(x) for x in '$REQ_CMAKE'.split('.')[:2] if x.isdigit()]
a += [0] * (2 - len(a)); b += [0] * (2 - len(b))
sys.exit(0 if a >= b else 1)"; then
  printf '  %-14s OK   %s\n' "cmake" "$cmake_ver"
else
  printf '  %-14s TOO OLD  %s (needs >=%s)\n' "cmake" "${cmake_ver:-?}" "$REQ_CMAKE"; fail=1
fi

log "result"
if [ "$fail" -eq 0 ]; then
  echo "ALL PREREQUISITES PRESENT - run: bash scripts/build.sh"
else
  echo "SOME PREREQUISITES MISSING (see above)."
  echo
  echo "If the FFmpeg floors failed, the distro is too old: use Ubuntu 24.04, which satisfies"
  echo "all floors from its own repositories (FFmpeg 6.1, libcurl 8.5, gcc-13 default)."
fi
exit "$fail"
