#!/usr/bin/env bash
# ============================================================
# ninfer-l20 : follow an upstream update.
#
# This is the workflow the original port did not have: a plain
# unified diff applied to a moving tree. Here the port is a set
# of re-runnable changes (patches with three-way fallback, a
# variant ladder of known source shapes, and a tree-derived
# sweep), so "upstream moved" reduces to:
#
#   1. fetch the upstream tip,
#   2. reset the port branch onto it,
#   3. re-apply the port (idempotent engine),
#   4. re-verify the invariants,
#   5. re-record the patches against the new base (keeps the
#      plain 2-way path valid for the next run),
#   6. rebuild incrementally.
#
# If an upstream change outruns every known shape, the engine
# reports the exact drift (file, expected anchor, current
# context) and verify names the invariant that regressed -
# nothing fails inside a 1400-file CUDA build first.
#
# Usage:
#   bash scripts/upgrade.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NINFER_L20_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SRC="$ROOT/src"
PYTHON="${PYTHON:-python3}"

step() { printf '\n########## %s ##########\n' "$*"; }
die()  { printf 'FATAL: %s\n' "$*" >&2; exit 1; }

[ -f "$ROOT/port/manifest.json" ] || die "manifest not found (is this a ninfer-l20 checkout?)"
[ -d "$SRC/.git" ] || die "no upstream checkout at $SRC (run: bash scripts/build.sh first)"

eval "$("$PYTHON" - "$ROOT/port/manifest.json" <<'EOF'
import json, shlex, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
print("UPSTREAM_BRANCH=" + shlex.quote(m["base"]["branch"]))
print("PORT_BRANCH=" + shlex.quote(m["layout"]["port_branch"]))
print("PORT_PREFIX=" + shlex.quote(m["layout"]["port_commit_prefix"]))
EOF
)"

step "upstream"
git -C "$SRC" fetch --quiet origin "$UPSTREAM_BRANCH"
NEW_TIP="$(git -C "$SRC" rev-parse "origin/$UPSTREAM_BRANCH")"
OLD_HEAD="$(git -C "$SRC" rev-parse HEAD)"

if [ "$OLD_HEAD" = "$NEW_TIP" ]; then
  echo "  already at the upstream tip - nothing to upgrade"
  exit 0
fi
printf '  upstream moves: %s -> %s (%s -> %s)\n' \
  "$(git -C "$SRC" log -1 --format=%h "$OLD_HEAD" 2>/dev/null || echo '?')" \
  "$(git -C "$SRC" log -1 --format=%h "$NEW_TIP")" \
  "$(git -C "$SRC" log -1 --format=%ad --date=short "$OLD_HEAD" 2>/dev/null || echo '?')" \
  "$(git -C "$SRC" log -1 --format=%ad --date=short "$NEW_TIP")"

# the port branch must be at a port commit (or the bare tip) - refuse a dirty tree
DIRTY="$(git -C "$SRC" status --porcelain | grep -v '^??' || true)"
[ -z "$DIRTY" ] || die "worktree has uncommitted changes - commit or stash them, then re-run"

OLD_PORT_COMMIT="$(git -C "$SRC" log --format=%H -1 --grep="$PORT_PREFIX" || true)"
if [ -n "$OLD_PORT_COMMIT" ] && git -C "$SRC" rev-parse --verify -q "${OLD_PORT_COMMIT}^" >/dev/null; then
  OLD_PORT_BASE="$(git -C "$SRC" rev-parse "${OLD_PORT_COMMIT}^")"
else
  OLD_PORT_BASE="$OLD_HEAD"
fi

# --- what upstream changed in the sources ------------------------------
step "upstream changes (src/ only)"
git -C "$SRC" diff --stat "origin/$UPSTREAM_BRANCH" "$NEW_TIP" -- src 2>/dev/null | tail -5 || true
git -C "$SRC" log --oneline "${OLD_PORT_BASE}..$NEW_TIP" -- src 2>/dev/null | head -15 || true

# --- reset the port branch onto the new tip, re-apply ------------------
step "re-apply the port on the new base"
git -C "$SRC" checkout -q -B "$PORT_BRANCH" "origin/$UPSTREAM_BRANCH"
"$PYTHON" "$ROOT/tools/port_engine.py" --worktree "$SRC"
NEW_PORT_COMMIT="$(git -C "$SRC" rev-parse HEAD)"

step "verify port invariants"
"$PYTHON" "$ROOT/tools/verify.py" --worktree "$SRC"

step "re-record the change patches against the new base"
# keeps the plain 2-way apply path valid for future runs; the regenerated files
# belong in the port repo - commit them there if this checkout is one.
"$PYTHON" "$ROOT/tools/regen_patches.py" --worktree "$SRC"
if git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  (cd "$ROOT" && git add port/changes/ 2>/dev/null && \
   { git diff --cached --quiet || git commit -qm "port: re-record change patches on upstream $(git -C "$SRC" rev-parse --short "origin/$UPSTREAM_BRANCH")"; } ) \
  && echo "  port repo: patch set re-based" || echo "  port repo: no patch changes"
fi

# --- rebuild incrementally ---------------------------------------------
step "rebuild"
if [ -d "$SRC/build" ] && command -v cmake >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1; then
  cmake --build "$SRC/build" -j
  echo "  serve: bash scripts/start.sh [port]"
else
  echo "  (build dir or toolchain absent - run: bash scripts/build.sh to do a full configure+build)"
fi

step "upgrade summary"
NEW_BASE="$(git -C "$SRC" rev-parse "origin/$UPSTREAM_BRANCH")"
if [ -n "$OLD_PORT_COMMIT" ]; then
  printf '  port before: %s\n' "$(git -C "$SRC" log -1 --format='%h %s' "$OLD_PORT_COMMIT")"
  printf '  port after:  %s\n' "$(git -C "$SRC" log -1 --format='%h %s' "$NEW_PORT_COMMIT")"
  echo "  port surface before (old base -> old port):"
  git -C "$SRC" diff --stat "$OLD_PORT_BASE" "$OLD_PORT_COMMIT" -- src | tail -3 | sed 's/^/    /'
  echo "  port surface after (new base -> new port):"
  git -C "$SRC" diff --stat "$NEW_BASE" "$NEW_PORT_COMMIT" -- src | tail -3 | sed 's/^/    /'
fi
echo "  upstream base is now $(git -C "$SRC" rev-parse --short "origin/$UPSTREAM_BRANCH") - provenance: bash scripts/build.sh rewrites build-info.json"
