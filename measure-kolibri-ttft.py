#!/usr/bin/env python3
"""Measure time to first token for Kolibri 1, cold and on an exact-APC hit.

The point of this script is issue #8. The Kolibri card reports decode rates
only, and on Apple Silicon at long context the prefill is what the user waits
for. Decode at 96k is 40 t/s; the prefill in front of it is a minute and a half.
This script measures the wait, so the card can state it.

    ./measure-kolibri-ttft.py                                  # 1k..96k, 2 repeats
    ./measure-kolibri-ttft.py --contexts 1000,8000 --repeat 1  # quick check

WHAT THE ARMS ARE. Per context and repeat, two requests:

    cold  a fresh nonce in front of the prompt, so no snapshot can match
    warm  the cold conversation, its answer, and one new question: the
          follow-up turn of a chat, served from the exact-APC snapshot

TTFT IS MEASURED ON THE CLIENT. The request streams; TTFT is the time from
sending the request to the first chunk that carries content, reasoning or a
tool call. The role chunk the server sends up front does not count. The
server's own `timings.prompt_ms` is recorded next to it; the two should agree
to within the HTTP and template overhead, and a large gap is reported, not
averaged away.

A cold arm with cache_n > 0 or a warm arm with cache_n == 0 is a broken
measurement and is flagged as such.

THE REASONING EFFORT DECIDES WHETHER A WARM HIT CAN EXIST. With
`reasoning_effort: none` Kolibri's template ends the generation prompt with an
empty `<think>\n\n</think>\n\n`, but renders past assistant turns without it.
The follow-up prompt therefore diverges from the stored snapshot 19 characters
before its end, and the sliding-window caches cannot be rewound to the point
of divergence: with `none`, every follow-up turn is a cold prefill. `low`,
`medium` and `high` keep the prefix stable. The default here is `low`, the
documented client setting; `--effort none` measures the miss.

The first request after a server start compiles the Metal kernels (25 s for a
1k prompt, measured); one discarded warm-up request goes first.

CAREFUL, THE SERVER IS THE MEASUREMENT. Versions, PREFILL_STEP and the APC
setting are read from the RUNNING server process (its command line, its
environment, its python), not from this shell. Start the server under
`caffeinate -dimsu` and with MEM_PROBE_INTERVAL=0.5, so the memory peak per
request has enough samples:

    MEM_PROBE_INTERVAL=0.5 caffeinate -dimsu ./start-mlx_kolibri.sh
"""

import argparse
import datetime
import importlib.util
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(name, alias):
    spec = importlib.util.spec_from_file_location(alias, os.path.join(_HERE, name))
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The filler pool and the log reader are part of the instrument; one source.
_mda = _load_sibling("measure-drafter-acceptance.py", "_mda")
_maw = _load_sibling("measure-apc-warm-decode.py", "_maw")

_FOLLOW_UP = "Summarize the log above in one sentence."


def _sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def server_context(port):
    """What the running server executes: python, versions, flags, APC, wired limit."""
    ctx = {
        "date": datetime.date.today().isoformat(),
        "chip": _sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "ram_gb": round(int(_sh("sysctl", "-n", "hw.memsize") or 0) / 2**30),
        "wired_limit_mb": int(_sh("sysctl", "-n", "iogpu.wired_limit_mb") or 0),
        "macos": platform.mac_ver()[0],
    }
    pid = _sh("lsof", "-tiTCP:%d" % port, "-sTCP:LISTEN").split("\n")[0]
    if not pid:
        return ctx
    cmd = _sh("ps", "-o", "command=", "-p", pid)
    env = _sh("ps", "eww", "-o", "command=", "-p", pid)

    def flag(name, default=None):
        m = re.search(r"--%s\s+(\S+)" % re.escape(name), cmd)
        return m.group(1) if m else default

    ctx.update(
        pid=int(pid),
        prefill_step=int(flag("prefill-step-size", 0)) or None,
        kv_bits=flag("kv-bits", "f16"),
        max_num_seqs=flag("max-num-seqs"),
        apc="APC_ENABLED=1" in env,
        apc_entries=(re.search(r"APC_EXACT_CACHE_ENTRIES=(\d+)", env) or [None, None])[1],
        apc_reserve_gb=(re.search(r"APC_MEMORY_RESERVE_GB=([\d.]+)", env) or [None, "auto"])[1],
    )
    python = cmd.split()[0] if cmd else ""
    if os.path.isfile(python):
        out = _sh(
            python,
            "-c",
            "import importlib.metadata as m;print(m.version('mlx'),m.version('mlx-vlm'))",
        ).split()
        if len(out) == 2:
            ctx["mlx"], ctx["mlx_vlm"] = out
    return ctx


def ask_stream(url, model, messages, max_tokens, timeout, effort):
    """One streamed chat request: TTFT on the client plus the server's usage/timings."""
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "reasoning_effort": effort,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    ttft, text, usage, timings = None, [], {}, {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning") or ""
                if ttft is None and (piece or delta.get("tool_calls")):
                    ttft = time.monotonic() - t0
                text.append(delta.get("content") or "")
            usage = chunk.get("usage") or usage
            timings = chunk.get("timings") or timings
    return {
        "ttft_s": ttft,
        "wall_s": time.monotonic() - t0,
        "answer": "".join(text),
        "usage": usage,
        "timings": timings,
    }


def run_arm(args, messages, window, arm, target):
    window.mark()
    r = ask_stream(args.url, args.model, messages, args.max_tokens, args.timeout, args.effort)
    mem_sum, _ = window.collect()
    t, u = r["timings"], r["usage"]
    prompt_tokens = u.get("prompt_tokens", 0)
    cache_n = t.get("cache_n", (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
    prompt_s = (t.get("prompt_ms") or 0) / 1000.0 or None
    broken = (arm == "cold" and cache_n > 0) or (arm == "warm" and cache_n == 0)
    return {
        "arm": arm,
        "target": target,
        "prompt_tokens": prompt_tokens,
        "prefilled": t.get("prompt_n", prompt_tokens - cache_n),
        "cache_n": cache_n,
        "ttft_s": r["ttft_s"],
        "server_prefill_s": prompt_s,
        "prefill_tps": t.get("prompt_per_second"),
        "wall_s": r["wall_s"],
        "mem_sum_gib": mem_sum,
        "broken": broken,
        "answer": r["answer"],
    }


def _fmt(v, spec="7.2f"):
    return "      -" if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="Kolibri-1-local")
    ap.add_argument(
        "--model-dir",
        default=os.path.expanduser("~/src/mlx/models/Kolibri-1-MLX-3bit"),
        help="tokenizer source for exact prompt lengths",
    )
    ap.add_argument("--contexts", default="1000,8000,32000,64000,96000")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--effort", default="low", choices=["none", "low", "medium", "high"])
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--log", default=os.path.expanduser("~/.mlx-kolibri/logs/server.log"))
    ap.add_argument(
        "--out",
        default=os.path.expanduser("~/.mlx-kolibri/measurements/ttft.jsonl"),
        help="results are appended here, one line per request",
    )
    args = ap.parse_args()

    port = int(args.url.rsplit(":", 1)[-1].split("/")[0])
    ctx = server_context(port)
    if "pid" not in ctx:
        sys.exit(f"no server listening on port {port}")
    print("server : " + ", ".join(f"{k}={v}" for k, v in ctx.items()))

    tok = _mda.load_tokenizer(args.model_dir)
    if tok is None:
        sys.exit("a tokenizer is required here; prompt lengths must be exact")
    window = _maw.LogWindow(args.log)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    targets = [int(x) for x in args.contexts.split(",")]
    run_id = uuid.uuid4().hex[:8]

    # Warm-up: kernel compilation, not prefill. Discarded.
    ask_stream(args.url, args.model, [{"role": "user", "content": f"Hi {run_id}"}],
               4, args.timeout, args.effort)
    ctx["effort"] = args.effort

    rows = []
    for rep in range(args.repeat):
        for target in targets:
            nonce = f"[run {run_id}-{rep}-{target}-{uuid.uuid4().hex}]\n"
            prompt = nonce + _mda.build_prompt(target - len(tok.encode(nonce)), tok)
            messages = [{"role": "user", "content": prompt}]
            cold = run_arm(args, messages, window, "cold", target)
            messages += [
                {"role": "assistant", "content": cold["answer"]},
                {"role": "user", "content": _FOLLOW_UP},
            ]
            warm = run_arm(args, messages, window, "warm", target)
            for r in (cold, warm):
                r.update(ctx, repeat=rep, run=run_id)
                r.pop("answer")
                rows.append(r)
                with open(args.out, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
                print(
                    f"  {r['arm']:4s} {target:6d}  prompt {r['prompt_tokens']:6d}"
                    f"  cached {r['cache_n']:6d}  TTFT {_fmt(r['ttft_s'])} s"
                    f"  server prefill {_fmt(r['server_prefill_s'])} s"
                    f"  mem {_fmt(r['mem_sum_gib'], '5.1f')} GiB"
                    + ("  BROKEN" if r["broken"] else ""),
                    flush=True,
                )

    print(
        f"\nKolibri 1 TTFT, {ctx.get('chip')}, {ctx.get('ram_gb')} GB, mlx {ctx.get('mlx')}, "
        f"mlx-vlm {ctx.get('mlx_vlm')}, PREFILL_STEP {ctx.get('prefill_step')}, "
        f"wired {ctx.get('wired_limit_mb')} MB, effort {args.effort}, {ctx['date']}\n"
    )
    print("| context | cold TTFT | prefill | warm TTFT | restored | peak mem |")
    print("|---|---|---|---|---|---|")
    for target in targets:
        good = [r for r in rows if r["target"] == target and not r["broken"]]
        cold = [r for r in good if r["arm"] == "cold" and r["ttft_s"]]
        warm = [r for r in good if r["arm"] == "warm" and r["ttft_s"]]
        if not cold:
            print(f"| {target} | broken | | | | |")
            continue
        c = statistics.median(r["ttft_s"] for r in cold)
        tps = statistics.median(r["prefill_tps"] or 0 for r in cold)
        w = statistics.median(r["ttft_s"] for r in warm) if warm else None
        restored = statistics.median(r["cache_n"] for r in warm) if warm else None
        mems = [r["mem_sum_gib"] for r in good if r["mem_sum_gib"]]
        print(
            f"| {cold[0]['prompt_tokens']:,} | {c:.1f} s | {tps:,.0f} t/s | "
            + (f"{w:.1f} s | {restored:,.0f} | " if w is not None else "broken | | ")
            + (f"{max(mems):.1f} GiB |" if mems else "- |")
        )
    spread = [
        (r["target"], r["arm"], r["ttft_s"] - r["server_prefill_s"])
        for r in rows
        if r["ttft_s"] and r["server_prefill_s"] and r["ttft_s"] - r["server_prefill_s"] > 1.0
    ]
    for target, arm, gap in spread:
        print(f"NOTE: {arm} {target}: TTFT exceeds the server prefill by {gap:.1f} s")
    print(f"\nraw: {args.out}")


if __name__ == "__main__":
    main()
