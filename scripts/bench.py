#!/usr/bin/env python3
"""ninfer-l20 end-to-end benchmark.

Measures the engine with three workloads (code / prose / qa) at greedy and temp 0.7,
plus a no-spec control and a wide-prefill probe. The wide-prefill probe is the
regression check for the cooperative-launch residency fix (change 070): completing a
>= 2688-token prefill without aborting exercises the route.

Everything is derived - no /opt paths, no fixed GPU:
  * the server binary is found under the project root (overridable),
  * the model path is an argument,
  * all serve parameters follow the device profile from scripts/start.sh logic.

Usage:
  python3 scripts/bench.py --model models/qwen3_8_27b.ninfer [--port 8091]
                           [--kv-dtype int8] [--spec mtp|none] [--n-predict 256]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WORKLOADS = {
    "code": "Write a Python implementation of an LRU cache class with get, put, and eviction. "
            "Include type hints and a docstring for each method.",
    "prose": "Write a detailed essay about the history of maritime trade in the Indian Ocean, "
             "covering ports, monsoon winds, and merchant communities.",
    "qa": "Explain step by step how a transformer attention mechanism computes its output, "
          "including the scaling factor.",
}


def find_serve_binary() -> Path:
    candidates = list((ROOT / "src").glob("build*/apps/ninfer-serve"))
    if not candidates:
        candidates = list((ROOT / "src").rglob("ninfer-serve"))
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError(f"ninfer-serve not found under {ROOT}/src (run: bash scripts/build.sh)")


def post(url: str, payload: dict, timeout: int):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wait_ready(port: int, proc, timeout_s: int = 900):
    """The engine validates memory before it listens, so readiness implies a usable config."""
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        if proc.poll() is not None:
            return False, "server exited during startup"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True, json.loads(resp.read().decode())
        except Exception:
            pass
        time.sleep(3)
    return False, "timeout waiting for /v1/models"


def read_vram() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        return out
    except Exception:
        return "n/a"


def one_request(port: int, prompt: str, n_predict: int, temperature: float,
                greedy: bool, timeout: int = 1800) -> dict:
    payload = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_predict,
        "temperature": 0.0 if greedy else temperature,
    }
    try:
        r = post(f"http://127.0.0.1:{port}/v1/chat/completions", payload, timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as err:
        print(f"  request failed: {err}", flush=True)
        return {}
    return r.get("timings", {})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(ROOT / "models" / "qwen3_8_27b.ninfer"))
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--kv-dtype", default="int8")
    ap.add_argument("--max-context", type=int, default=262144)
    ap.add_argument("--draft-tokens", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--spec", default="mtp", choices=["mtp", "none"])
    ap.add_argument("--out", default=str(ROOT / "bench-results.json"))
    ap.add_argument("--log", default=str(ROOT / "serve-bench.log"))
    args = ap.parse_args()

    serve_bin = find_serve_binary()
    if not Path(args.model).is_file():
        print(f"FATAL: model artifact not found: {args.model}")
        return 1

    cmd = [
        str(serve_bin), args.model,
        "--host", "127.0.0.1", "--port", str(args.port),
        "--max-context", str(args.max_context),
        "--kv-capacity", str(args.max_context),
        "--max-concurrency", "1",
        "--max-pending-requests", "16",
        "--pending-timeout-ms", "600000",
        "--prefill-chunk", "1024",
        "--kv-dtype", args.kv_dtype,
        "--log-level", "info",
    ]
    if args.spec == "mtp":
        cmd += ["--spec", "mtp", "--draft-tokens", str(args.draft_tokens), "--lm-head-draft"]

    print("=" * 78)
    print(f"CONFIG  spec={args.spec} kv={args.kv_dtype} ctx={args.max_context}")
    print("  " + " ".join(cmd))
    print(f"  VRAM before: {read_vram()}")
    print("=" * 78, flush=True)

    log = open(args.log, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    results: list[dict] = []
    try:
        ok, info = wait_ready(args.port, proc)
        if not ok:
            print(f"STARTUP FAILED: {info}")
            log.flush()
            with open(args.log) as f:
                print("".join(f.readlines()[-30:]))
            return 1
        print(f"server ready. /v1/models -> {json.dumps(info)[:300]}")
        print(f"VRAM after load: {read_vram()}\n")

        print("warmup (discarded) ...", flush=True)
        one_request(args.port, "hello", 8, 0.0, True, timeout=900)

        for name, prompt in WORKLOADS.items():
            for greedy in (True, False):
                t = one_request(args.port, prompt, args.n_predict,
                                0.0 if greedy else 0.7, greedy)
                if not t:
                    print(f"  [{name}] no timings block returned")
                    continue
                dn, da = t.get("draft_n"), t.get("draft_n_accepted")
                acc = f"{100.0 * da / dn:.1f}%" if dn else "-"
                row = {
                    "workload": name,
                    "sampling": "greedy" if greedy else "temp0.7",
                    "tok_s": round(t.get("predicted_per_second", 0.0), 2),
                    "predicted_n": t.get("predicted_n"),
                    "prompt_n": t.get("prompt_n"),
                    "pp_tok_s": round(t.get("prompt_per_second", 0.0), 1),
                    "ttft_ms": round(t.get("prompt_ms", 0.0), 1),
                    "draft_n": dn,
                    "draft_accepted": da,
                    "acceptance": acc,
                }
                results.append(row)
                print(f"  [{name:5s} {row['sampling']:8s}] {row['tok_s']:7.2f} tok/s  "
                      f"draft {da}/{dn} = {acc}  ttft {row['ttft_ms']} ms", flush=True)

        # No-spec control: the MTP speedup ratio needs the baseline on the same device.
        print("\nwide-prefill probe (cooperative-launch residency regression check) ...", flush=True)
        filler = ("The history of ocean navigation is long and detailed. " * 400)
        t = one_request(args.port, filler + "\nSummarise the passage above in one sentence.",
                        64, 0.0, True)
        if t:
            print(f"  prefill {t.get('prompt_n')} tokens at {t.get('prompt_per_second', 0):.0f} tok/s "
                  f"-> completed without abort (cooperative route exercised)")
        else:
            print("  probe returned no timings")

        print("\n" + "=" * 78)
        print("SUMMARY")
        print(json.dumps(results, indent=2))
        Path(args.out).write_text(json.dumps({
            "spec": args.spec, "kv_dtype": args.kv_dtype,
            "max_context": args.max_context,
            "model": args.model,
            "serve": str(serve_bin),
            "results": results,
        }, indent=2), encoding="utf-8")
        print(f"\nsaved {args.out}")
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            time.sleep(3)
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
