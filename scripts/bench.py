#!/usr/bin/env python3
"""Ninfer4L20 engine benchmark.

Spawns the serve binary with the device profile, then measures:
  * three workloads (code / prose / qa) at greedy and temp-0.7 sampling,
  * the MTP draft-acceptance rate per request,
  * a wide-prefill probe (long prompt, short generation). The probe is the
    regression check for the cooperative-launch residency fix: a prefill
    beyond the 2688-token chunk threshold exercises the split-8 cooperative
    route that the old hard-coded residency constant broke on low-SM parts.

Everything is derived at run time (no fixed install prefix):
  * the serve binary is located under the project root,
  * model path / port / profile knobs are arguments or NINFER_* env vars.

Usage:
  python3 scripts/bench.py --model <path.to.ninfer> [--port 8091]
                           [--spec mtp|none] [--tokens 256]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "qwen3_8_27b.ninfer"

# Workload prompts (self-authored; kept short so generation dominates wall time).
WORKLOADS: dict[str, str] = {
    "code": (
        "Implement a thread-safe token-bucket rate limiter in Python with type "
        "hints: acquire(tokens) blocks until capacity is available, and a "
        "refill thread keeps the bucket topped up at a fixed per-second rate."
    ),
    "prose": (
        "Write a historical essay about how canal engineering shaped the rise "
        "and fall of trading cities in medieval Europe. Cover at least three "
        "cities and the goods that moved through them."
    ),
    "qa": (
        "Explain step by step how a binary heap maintains its invariants during "
        "insertion and deletion, and why each operation is O(log n)."
    ),
}

# Probe filler: repeated to push the prompt past the 2688-token wide-prefill
# threshold, then a trivial generation request.
PROBE_FILLER = (
    "The construction of the Suez Canal required massive dredging campaigns, "
    "earthworks measured in millions of cubic metres, and a labour force drawn "
    "from across the Mediterranean. "
)
PROBE_QUESTION = "\nIn one sentence, state the main engineering challenge above."


def locate_serve() -> Path:
    hits = sorted((ROOT / "src").rglob("ninfer-serve"))
    for h in hits:
        if h.is_file() and os.access(h, os.X_OK):
            return h
    raise FileNotFoundError(
        f"no executable ninfer-serve found under {ROOT / 'src'} "
        "(run: bash scripts/build.sh)"
    )


def http_post(url: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class ServeProcess:
    """Lifecycle wrapper: start -> wait-ready -> chat -> stop."""

    def __init__(self, binary: Path, model: str, port: int, spec: str,
                 draft_tokens: int, kv_dtype: str, max_context: int,
                 log_path: Path):
        flags = [
            str(binary), model,
            "--host", "127.0.0.1", "--port", str(port),
            "--max-context", str(max_context),
            "--kv-capacity", str(max_context),
            "--max-concurrency", "1",
            "--max-pending-requests", "16",
            "--pending-timeout-ms", "600000",
            "--prefill-chunk", "1024",
            "--kv-dtype", kv_dtype,
            "--log-level", "info",
        ]
        if spec == "mtp":
            flags += ["--spec", "mtp", "--draft-tokens", str(draft_tokens),
                      "--lm-head-draft"]
        self.flags = flags
        self.port = port
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.log_file = None

    def start(self, startup_timeout: int = 600) -> dict:
        self.log_file = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            self.flags, stdout=self.log_file, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        # The engine validates its memory budget before it binds the socket:
        # a 200 from /v1/models implies a usable configuration.
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"server exited during startup (rc={self.proc.returncode})"
                )
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{self.port}/v1/models", timeout=5) as r:
                    if r.status == 200:
                        return json.loads(r.read().decode("utf-8"))
            except (urllib.error.URLError, urllib.error.HTTPError,
                    TimeoutError, ConnectionError):
                pass
            time.sleep(3)
        raise RuntimeError("server did not become ready within the timeout")

    def chat(self, prompt: str, max_tokens: int, temperature: float,
             timeout: int = 1800) -> dict | None:
        """Returns the engine timings block, or None on failure."""
        body = {
            "model": "qwen3.8-27b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            data = http_post(
                f"http://127.0.0.1:{self.port}/v1/chat/completions", body, timeout
            )
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, ConnectionError) as err:
            print(f"  request failed: {err}", flush=True)
            return None
        return data.get("timings")

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(25):  # up to 5s for a clean shutdown
            if self.proc.poll() is not None:
                break
            time.sleep(0.2)
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.proc.wait()
        if self.log_file:
            self.log_file.close()


def gpu_memory_line() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        return out or "n/a"
    except (OSError, subprocess.SubprocessError):
        return "n/a"


def fmt_acceptance(timings: dict) -> str:
    drafted = timings.get("draft_n") or 0
    accepted = timings.get("draft_n_accepted") or 0
    if not drafted:
        return "-"
    return f"{accepted}/{drafted} ({100.0 * accepted / drafted:.1f}%)"


def run_suite(args) -> int:
    binary = locate_serve()
    if not Path(args.model).is_file():
        print(f"FATAL: model artifact not found: {args.model}")
        return 1

    print("=" * 76)
    print(f"spec={args.spec}  kv={args.kv_dtype}  ctx={args.max_context}  "
          f"tokens={args.tokens}")
    print(f"binary: {binary}")
    print(f"model:  {args.model}")
    print(f"VRAM:   {gpu_memory_line()}")
    print("=" * 76, flush=True)

    server = ServeProcess(
        binary, args.model, args.port, args.spec, args.draft_tokens,
        args.kv_dtype, args.max_context, Path(args.log),
    )
    records: list[dict] = []
    probe_record: dict | None = None
    try:
        models = server.start()
        print(f"ready. /v1/models -> {json.dumps(models)[:240]}")
        print(f"VRAM:   {gpu_memory_line()}\n")

        # one discarded warm-up request (JIT/CUDA-graph paths settle)
        server.chat("hello", 8, 0.0, timeout=900)
        print("warm-up done\n", flush=True)

        for name, prompt in WORKLOADS.items():
            for label, temp in (("greedy", 0.0), ("temp0.7", 0.7)):
                t = server.chat(prompt, args.tokens, temp)
                if t is None:
                    print(f"  [{name}/{label}] no timings returned")
                    continue
                row = {
                    "workload": name,
                    "sampling": label,
                    "tokens_per_s": round(t.get("predicted_per_second", 0.0), 2),
                    "generated": t.get("predicted_n"),
                    "prompt_tokens": t.get("prompt_n"),
                    "prefill_tok_s": round(t.get("prompt_per_second", 0.0), 1),
                    "ttft_ms": round(t.get("prompt_ms", 0.0), 1),
                    "mtp": fmt_acceptance(t),
                }
                records.append(row)
                print(
                    f"  [{name:5s}/{label:7s}] {row['tokens_per_s']:7.2f} tok/s  "
                    f"mtp={row['mtp']}  ttft={row['ttft_ms']}ms",
                    flush=True,
                )

        print("\nwide-prefill probe (cooperative-launch regression) ...", flush=True)
        filler = PROBE_FILLER * 400  # ~13-14k tokens: well past the 2688 threshold
        t = server.chat(filler + PROBE_QUESTION, 48, 0.0)
        if t:
            probe_record = {
                "prompt_tokens": t.get("prompt_n"),
                "prefill_tok_s": round(t.get("prompt_per_second", 0.0), 1),
                "completed_without_abort": True,
            }
            print(f"  prefill {t.get('prompt_n')} tokens "
                  f"@ {t.get('prompt_per_second', 0.0):.0f} tok/s - OK")
        else:
            probe_record = {"completed_without_abort": False}
            print("  probe FAILED (no timings returned)")
    finally:
        server.stop()

    print("\n" + "=" * 76)
    for r in records:
        print(
            f"  {r['workload']:5s} {r['sampling']:7s} "
            f"{r['tokens_per_s']:7.2f} tok/s  mtp={r['mtp']}"
        )
    print("=" * 76)

    out = {
        "config": {
            "spec": args.spec,
            "kv_dtype": args.kv_dtype,
            "max_context": args.max_context,
            "draft_tokens": args.draft_tokens if args.spec == "mtp" else None,
            "tokens_per_request": args.tokens,
            "model": args.model,
            "serve_binary": str(binary),
        },
        "results": records,
        "probe": probe_record,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("NINFER_PORT", "8091")))
    ap.add_argument("--spec", default="mtp", choices=["mtp", "none"])
    ap.add_argument("--draft-tokens", type=int, default=3)
    ap.add_argument("--kv-dtype", default="int8")
    ap.add_argument("--max-context", type=int, default=262144)
    ap.add_argument("--tokens", type=int, default=256,
                    help="generated tokens per benchmark request")
    ap.add_argument("--out", default=str(ROOT / "bench-results.json"))
    ap.add_argument("--log", default=str(ROOT / "serve-bench.log"))
    args = ap.parse_args()
    return run_suite(args)


if __name__ == "__main__":
    sys.exit(main())
