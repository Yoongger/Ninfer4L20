#!/usr/bin/env bash
# ============================================================
# Ninfer4L20 : fetch the Qwen3.8-27B NInfer artifact with a
# pinned revision + checksum.
#
# Why pin: HF's `main` moved to a v3 container on 2026-09-15;
# this fork's baseline reads v2. Without a pin the server dies
# at startup with "artifact magic is not NInfer v2". The pinned
# revision is the initial v2 artifact (16.96 GiB).
#
# Usage:
#   bash scripts/fetch-model.sh
#   NINFER_MODEL_REVISION=<sha> bash scripts/fetch-model.sh
#
# Override the endpoint for mainland-China networks:
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/fetch-model.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NINFER_L20_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OUT_DIR="${NINFER_MODEL_DIR:-$ROOT/models}"
REVISION="${NINFER_MODEL_REVISION:-3526913004}"
ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
REPO="neroued/Qwen3.8-27B-NInfer"
FILE="qwen3_8_27b.ninfer"
OUT="$OUT_DIR/$FILE"
EXPECT_SHA256="eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e"

mkdir -p "$OUT_DIR"
BASE="$ENDPOINT/$REPO/resolve/$REVISION"

echo "  revision: $REVISION"
echo "  endpoint: $ENDPOINT"
echo "  target:   $OUT"

# header-only container-version probe: 8 bytes, no full download
if [ ! -s "$OUT" ]; then
  echo "  probing artifact magic (first 8 bytes)..."
  head8=$(curl -fsSL -r 0-7 "$BASE/$FILE" | od -An -tx1 | tr -d ' \n')
  echo "  magic bytes: $head8"
  case "$head8" in
    "" ) echo "  [warn] empty probe response";;
  esac
fi

if [ -s "$OUT" ] && [ "$(wc -c < "$OUT")" -gt 1000000000 ]; then
  echo "  existing artifact present, resuming only if incomplete"
fi

# download (resumable) + checksum manifest
curl -fsSL -o "$OUT_DIR/SHA256SUMS" "$BASE/SHA256SUMS" || echo "  [warn] could not fetch SHA256SUMS; verifying against the pinned digest below"
curl -L -C - --retry 8 --retry-all-errors -o "$OUT" "$BASE/$FILE"

if [ -f "$OUT_DIR/SHA256SUMS" ] && grep -q "$FILE" "$OUT_DIR/SHA256SUMS"; then
  (cd "$OUT_DIR" && sha256sum -c SHA256SUMS) || {
    echo "FATAL: checksum mismatch - delete $OUT and re-run"
    exit 1
  }
elif [ -n "$EXPECT_SHA256" ]; then
  actual=$(sha256sum "$OUT" | cut -d' ' -f1)
  [ "$actual" = "$EXPECT_SHA256" ] || { echo "FATAL: checksum mismatch: $actual"; exit 1; }
  echo "  checksum OK (pinned digest)"
fi

echo "done: $OUT ($(du -h "$OUT" | cut -f1))"
