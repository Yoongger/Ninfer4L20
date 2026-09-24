#!/usr/bin/env bash
# ============================================================
# Ninfer4L20 : start the engine with a profile chosen from the
# GPU that is actually present.
#
# The original port hard-coded the L20 profile (262144 context,
# INT8 KV, MTP k3) into the script. Here the profile is selected
# at launch time from nvidia-smi, and every parameter can be
# overridden with an environment variable:
#
#   NINFER_PORT            (default 8090)
#   NINFER_HOST            (default 0.0.0.0)
#   NINFER_MODEL           (default $ROOT/models/qwen3_8_27b.ninfer)
#   NINFER_MAX_CONTEXT     NINFER_KV_CAPACITY
#   NINFER_KV_DTYPE        int8 | bf16 | fp8 | ...   (default per profile)
#   NINFER_SPEC            mtp | none        (default mtp)
#   NINFER_DRAFT_TOKENS    (default 3)
#   NINFER_PREFILL_CHUNK   (default per profile; keep <= 2688, see PORT-SPEC)
#   NINFER_MAX_CONCURRENCY (default 1)
#
# The L20 profile ships the fastest measured configuration
# (2026-09-24, L20 48 GB, 262144 context, same-build A/B):
#   --kv-dtype bf16      +3.6..12.8% decode vs int8, +1..3.4% prefill,
#                        highest KV fidelity (no quantisation)
#   --prefill-chunk 2688 +3.7..6.7% prefill vs 1024 at 86k..219k context
#   --spec mtp k3        fastest of draft-tokens 1..5 on both 27B artifacts
#   --max-concurrency 1  single-stream serving; conc 2 is -0.7% per request
#
# Usage:
#   bash scripts/start.sh [port]
# ============================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NINFER_L20_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# the upstream tree nests the build dir (src/build) inside the src/ subdir,
# so the binary lives ~4 levels down; search the whole worktree.
BIN="$(find "$ROOT/src" -type f -name ninfer-serve -executable 2>/dev/null | head -1)"
MODEL="${NINFER_MODEL:-$ROOT/models/qwen3_8_27b.ninfer}"
LOG="$ROOT/serve.log"
PORT="${1:-${NINFER_PORT:-8090}}"

[ -n "$BIN" ] && [ -x "$BIN" ] || { echo "FATAL: ninfer-serve not found under $ROOT/src (run: bash scripts/build.sh)"; exit 1; }
[ -f "$MODEL" ] || { echo "FATAL: model artifact not found: $MODEL (run: bash scripts/fetch-model.sh)"; exit 1; }

# --- pick a profile from the live device --------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FATAL: nvidia-smi not found - cannot select a device profile"
  exit 1
fi
GPU_LINE="$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits | awk 'NR==1')"
GPU_NAME="$(printf '%s' "$GPU_LINE" | cut -d, -f1 | tr -d ' ')"
GPU_MEM="$(printf '%s' "$GPU_LINE" | cut -d, -f2 | tr -d ' ')"   # MiB
printf '  GPU: %s (%s MiB)\n' "$GPU_NAME" "$GPU_MEM"

case "$GPU_NAME" in
  *L20*)
    # 48 GB: full native context with BF16 KV - the fastest measured
    # configuration (same-build A/B, 2026-09-24): +3.6..12.8% decode vs
    # int8 via higher MTP acceptance, +1..3.4% prefill, no KV
    # quantisation error. BF16 KV at 262144 is 16.5 GiB on device; the
    # engine fits the whole plan in 48 GB with room to spare.
    P_MAX_CONTEXT=262144; P_KV_CAPACITY=262144; P_KV_DTYPE=bf16
    P_PREFILL_CHUNK=2688; PROFILE="L20 (fastest measured)" ;;
  *"4090"*)
    # 24 GB: INT8 KV ceiling is ~172032 (196608 is rejected at startup).
    # BF16 KV would need 11.5 GiB of cache and does not fit the plan.
    P_MAX_CONTEXT=262144; P_KV_CAPACITY=172032; P_KV_DTYPE=int8
    P_PREFILL_CHUNK=1024; PROFILE="RTX 4090" ;;
  *)
    # Unknown sm_89 part: conservative KV, everything else identical.
    P_MAX_CONTEXT=262144; P_KV_CAPACITY=65536; P_KV_DTYPE=int8
    P_PREFILL_CHUNK=1024; PROFILE="unknown (conservative KV)" ;;
esac
printf '  profile: %s\n' "$PROFILE"

MAX_CONTEXT="${NINFER_MAX_CONTEXT:-$P_MAX_CONTEXT}"
KV_CAPACITY="${NINFER_KV_CAPACITY:-$P_KV_CAPACITY}"
KV_DTYPE="${NINFER_KV_DTYPE:-$P_KV_DTYPE}"
SPEC="${NINFER_SPEC:-mtp}"
DRAFT_TOKENS="${NINFER_DRAFT_TOKENS:-3}"
PREFILL_CHUNK="${NINFER_PREFILL_CHUNK:-$P_PREFILL_CHUNK}"
MAX_CONCURRENCY="${NINFER_MAX_CONCURRENCY:-1}"

# --- sanity: will the KV budget fit in the free VRAM? --------------------
KV_BYTES_PER_TOKEN=$(case "$KV_DTYPE" in
  int8) echo 35900 ;;
  f16|bf16) echo 68000 ;;   # bf16 KV measured 33792*2 B/token at 262144
  *)    echo 26400 ;;
esac)
USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
FREE=$((GPU_MEM - USED))
NEED_KB=$((KV_CAPACITY * KV_BYTES_PER_TOKEN / 1024))
FREE_KB=$((FREE * 1024))
if [ "$NEED_KB" -gt $((FREE_KB * 85 / 100)) ]; then
  echo "  [warn] KV budget ~$((NEED_KB / 1024)) GiB vs ~$((FREE_KB / 1024 / 1024)) GiB free VRAM -"
  echo "         the engine will reject the startup; lower NINFER_KV_CAPACITY or NINFER_KV_DTYPE"
fi

# free the port and the VRAM from any previous instance
pkill -f ninfer-serve 2>/dev/null
sleep 2

: > "$LOG"
cmd=( "$BIN" "$MODEL"
  --host "${NINFER_HOST:-0.0.0.0}" --port "$PORT"
  --max-context "$MAX_CONTEXT" --kv-capacity "$KV_CAPACITY"
  --max-concurrency "$MAX_CONCURRENCY" --max-pending-requests 16 --pending-timeout-ms 600000
  --prefill-chunk "$PREFILL_CHUNK" --kv-dtype "$KV_DTYPE"
  --log-level info )
if [ "$SPEC" = "mtp" ]; then
  cmd+=( --spec mtp --draft-tokens "$DRAFT_TOKENS" --lm-head-draft )
fi

printf '  launching: %s\n' "${cmd[*]}"
setsid nohup "${cmd[@]}" >> "$LOG" 2>&1 < /dev/null &
PID=$!
disown 2>/dev/null || true

# the engine validates its memory budget before it listens: a 200 from
# /v1/models therefore implies a usable configuration. Poll on a deadline.
READY_DEADLINE=$((SECONDS + 300))
while [ "$SECONDS" -lt "$READY_DEADLINE" ]; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "FAILED: server exited during startup. Last log lines:"
    tail -n 20 "$LOG"
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" -o /dev/null 2>/dev/null; then
    echo "ready after ~${SECONDS}s (pid=$PID, log=$LOG)"
    grep -iE 'sm|multiprocessor' "$LOG" 2>/dev/null | head -n 1
    exit 0
  fi
  sleep 3
done
echo "TIMEOUT: no readiness within ${READY_DEADLINE}s. Last log lines:"
tail -n 20 "$LOG"
exit 1
