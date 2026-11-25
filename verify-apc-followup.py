#!/usr/bin/env python3
"""Check that a follow-up turn on a long conversation is admitted and served from the prefix cache.

This is the agent-session shape: a long prompt, an answer, and the same
conversation again a moment later with a few thousand tokens appended. The
second request is the cheap one -- everything but the tail is in the exact-APC
snapshot -- and it is the one the server refused on 2026-11-21. The code path
behind that arrived with the move to 0.7.12 on 2026-11-06.

WHY THIS EXISTS. mlx-vlm 0.7.12 (#2645) turned a prefill that fails the APC
memory admission from a cold prefill into HTTP 500 ("The request cannot fit the
APC prefill memory budget even alone"). The admission read free+inactive from
vm_stat, which loses the server's own weights the moment the GPU runs, and it
ran AFTER the snapshot restore. So cold requests of any size passed and
follow-up turns past roughly 44-48k tokens were rejected (patch 0016, see
patches/apply-patches.sh). The upgrade checks we had -- test suites, the greedy
hash, cold/warm timing at 28k -- could not see it: none of them sends a second
turn at that size. This one does, and it takes about five minutes.

    ./verify-apc-followup.py                       # 44k -> 48k, then 60k -> 64k
    ./verify-apc-followup.py --sizes 30000         # one pair
    ./verify-apc-followup.py --grow 2000 --max-tokens 400

Exit status 0 when every pair passed, 1 otherwise.

WHAT A PAIR IS, AND WHAT PASSES.

    turn 1  a fresh nonce + SIZE tokens of filler       cold, cache_n == 0
    turn 2  the same messages + the answer + GROW more  must come back 200,
            tokens, sent immediately                    with cache_n within 64
                                                        tokens of turn 1's prompt

Turn 2 follows turn 1 without a pause on purpose: the weights are still wired
then, which is the state the admission has to cope with. A turn 2 that returns
200 with a small cache_n is also a failure -- that is the pre-0.7.12 behaviour,
a silent cold prefill of the whole conversation.

DO NOT POINT THIS AT THE PRODUCTION SERVER. Every turn stores a snapshot of
64 KiB per token; the default run writes about 14 GB into the SSD tier and
evicts real conversations to make room. The default --url is therefore port
8898, not 8888. Start a throwaway server next to it (stop the production one
first: two copies of the weights do not fit), and remove its state afterwards:

    STATE_DIR=/tmp/apc-followup PORT=8898 APC_DISK_MAX_GB=16 ./start-mlx_qwen3.8.sh
    ./verify-apc-followup.py --log /tmp/apc-followup/logs/server.log
    rm -rf /tmp/apc-followup

MEMORY (--log). With the server log given, each row carries the max of
active / cache / active+cache that the server's sampler logged during that
request. The sampler runs every MEM_PROBE_INTERVAL seconds (default 5); a
follow-up turn lasts 15-20 s, so start the server with MEM_PROBE_INTERVAL=1 when
the memory columns matter. `peak` is mlx's process-lifetime peak and is never
reset.

The 60k -> 64k pair is the client context window. Without a cap on mlx's
buffer cache it took this 48 GB machine to its edge (2026-11-21: wired
40.2-40.5 GiB, swap +4 GB); with the start script's default
MLX_CACHE_LIMIT_GB=2 wired peaked at 33.5 GiB. One run takes about six minutes
(44k cold 110-116 s, 60k cold 167-170 s, follow-ups 25-29 s).

TIMINGS ARE ONLY WORTH READING ON A CHARGED BATTERY. A small adapter does not
cover this load: on 2026-11-21 the charge fell from 19% to 7% within half an
hour on mains power, and midway through the prefill dropped to half its rate
and stayed there. The pass/fail result does not depend on that.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    """Sibling scripts have hyphens in their names, so import them by path."""
    path = os.path.join(_HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# One filler pool and one log reader for all measure scripts.
_mda = _load("_mda", "measure-drafter-acceptance.py")
_warm = _load("_warm", "measure-apc-warm-decode.py")

_ASK_OK = "\n\nReply with the single word OK."
# Tolerance between turn 1's prompt and what turn 2 gets back from the cache.
# Measured: the snapshot comes back 1 token short of the prompt.
_CACHE_SLACK = 64


def filler(tokens, tokenizer, offset=0):
    """About *tokens* tokens of filler, exact when a tokenizer is given.

    *offset* shifts the start inside the pool so that the appended text of
    turn 2 is not a repetition of what turn 1 ended on.
    """
    pool = _mda._FILLER_POOL
    n = len(pool)
    if tokenizer is None:
        # 1.23 tokens per word, measured on this pool (measure-drafter-acceptance.py).
        words_needed = max(1, int(tokens / 1.23))
        sentences = words_needed // 12 + n
        words = " ".join(
            pool[(offset + i * 5) % n] for i in range(sentences)
        ).split()[:words_needed]
        return " ".join(words)
    text = " ".join(pool[(offset + i * 5) % n] for i in range(tokens // 4 + n))
    ids = tokenizer.encode(text, add_special_tokens=False)[:tokens]
    return tokenizer.decode(ids)


def chat(args, messages, max_tokens):
    """One request. Returns a dict with status, wall time and the usage figures."""
    body = {
        "model": args.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    req = urllib.request.Request(
        f"{args.url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    out = {"status": None, "error": None}
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            data = json.loads(resp.read())
            out["status"] = resp.status
    except urllib.error.HTTPError as exc:
        out["status"] = exc.code
        out["error"] = exc.read().decode("utf-8", "replace")[:400]
        out["wall_s"] = time.monotonic() - t0
        return out
    out["wall_s"] = time.monotonic() - t0
    t = data.get("timings") or {}
    usage = data.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    cache_n = t.get("cache_n", details.get("cached_tokens", 0)) or 0
    total = usage.get("prompt_tokens")
    if total is None:
        # timings.prompt_n is only what still had to be prefilled.
        total = (t.get("prompt_n") or 0) + cache_n
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    out.update(
        prompt_tokens=total,
        cache_n=cache_n,
        prefill_s=(t.get("prompt_ms") or 0) / 1000.0 or None,
        out_n=t.get("predicted_n", usage.get("completion_tokens", 0)),
        decode_tps=t.get("predicted_per_second"),
        content=msg.get("content") or "",
    )
    return out


def _row(pair, turn, r, mem):
    f = _warm._fmt
    if r["error"] is not None:
        return (
            f"  {pair:>4} {turn:>4}   HTTP {r['status']} after {r['wall_s']:.1f} s: "
            f"{r['error']}"
        )
    active, cache, total, peak = mem
    return (
        f"  {pair:>4} {turn:>4} {r['prompt_tokens']:>8} {r['cache_n']:>8} "
        f"{r['wall_s']:>8.1f} {f(r['prefill_s'], '9.1f', '        -')} "
        f"{r['out_n']:>6} {f(r['decode_tps'], '10.1f', '         -')} "
        f"{f(active, '7.2f', '      -')} {f(cache, '6.2f', '     -')} "
        f"{f(total, '6.2f', '     -')} {f(peak, '6.2f', '     -')}"
    )


def _mem(window):
    """(max active, max cache, max sum, last peak) in GiB over the marked window."""
    phases = window.collect_phases(float("inf"))  # everything lands in "prefill"
    _, peak = window.collect()
    if "prefill" not in phases:
        return None, None, None, peak
    active, cache, total, _n = phases["prefill"]
    return active, cache, total, peak


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--url",
        default="http://127.0.0.1:8898",
        help="throwaway server; NOT the production port, see the module text",
    )
    ap.add_argument("--model", default="Qwen3.8-27B-local")
    ap.add_argument(
        "--model-dir",
        default=os.path.expanduser("~/src/mlx/models/Qwen3.8-27B-MLX-4bit"),
        help='tokenizer source for exact prompt lengths; "" to skip',
    )
    ap.add_argument(
        "--sizes",
        default="44000,60000",
        help="turn-1 filler tokens, one pair each (comma-separated)",
    )
    ap.add_argument(
        "--grow", type=int, default=4000, help="filler tokens appended in turn 2"
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="decoded in turn 2 (turn 1 always asks for one word)",
    )
    ap.add_argument(
        "--reasoning-effort",
        default="none",
        help='sent with every request; "" to leave it to the server',
    )
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument(
        "--log", default="", help="server log, for the memory columns; empty skips them"
    )
    ap.add_argument("--save", default="", help="append one JSON line per turn here")
    ap.add_argument("--label", default="", help="free text stored with --save")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    tokenizer = _mda.load_tokenizer(args.model_dir)
    window = _warm.LogWindow(args.log) if args.log else _warm.LogWindow(os.devnull)

    print(f"  endpoint : {args.url}  model={args.model}")
    print(
        f"  pairs    : {', '.join(f'{s} -> +{args.grow}' for s in sizes)} tokens of "
        f"filler, {args.max_tokens} decoded in turn 2, temperature 0"
    )
    print(
        f"  lengths  : {'exact (tokenizer)' if tokenizer else 'estimated (no tokenizer)'}"
        f"; the chat template adds a few tokens on top\n"
    )
    print(
        f"  {'pair':>4} {'turn':>4} {'prompt':>8} {'cached':>8} {'wall_s':>8} "
        f"{'prefill_s':>9} {'out':>6} {'decode t/s':>10} {'active':>7} {'cache':>6} "
        f"{'sum':>6} {'peak':>6}"
    )
    print("  " + "-" * 94)

    failures = []
    records = []
    for pair, size in enumerate(sizes, 1):
        nonce = f"[run {time.time_ns()}] "
        messages = [
            {"role": "user", "content": nonce + filler(size, tokenizer) + _ASK_OK}
        ]
        try:
            window.mark()
            first = chat(args, messages, 8)
            mem1 = _mem(window)
            print(_row(pair, 1, first, mem1))
            records.append((pair, 1, size, first, mem1))
            if first["error"] is not None:
                failures.append(f"pair {pair}: turn 1 ({size}) returned HTTP {first['status']}")
                continue
            if first["cache_n"]:
                print(
                    f"       ^ turn 1 hit the cache (cached={first['cache_n']}); the "
                    f"timings of this pair are not cold/warm."
                )

            messages = messages + [
                {"role": "assistant", "content": first["content"].strip() or "OK"},
                {
                    "role": "user",
                    "content": filler(args.grow, tokenizer, offset=7) + _mda._QUESTION,
                },
            ]
            window.mark()
            second = chat(args, messages, args.max_tokens)
            mem2 = _mem(window)
            print(_row(pair, 2, second, mem2))
            records.append((pair, 2, size, second, mem2))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            # Refused, reset or timed out: the server is gone or hung, so the
            # remaining pairs would only repeat the same line.
            print(f"  {pair:>4}        request failed: {exc}")
            failures.append(f"pair {pair}: no answer from {args.url} ({exc})")
            break

        if second["error"] is not None:
            failures.append(
                f"pair {pair}: follow-up to {first['prompt_tokens']} tokens returned "
                f"HTTP {second['status']}"
            )
        elif second["cache_n"] < first["prompt_tokens"] - _CACHE_SLACK:
            failures.append(
                f"pair {pair}: follow-up restored {second['cache_n']} of "
                f"{first['prompt_tokens']} tokens -- the conversation was prefilled "
                f"again instead of served from the snapshot"
            )

    if args.save:
        with open(args.save, "a") as fh:
            for pair, turn, size, r, mem in records:
                rec = {
                    "label": args.label,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "pair": pair,
                    "turn": turn,
                    "size": size,
                    "grow": args.grow,
                    **{k: v for k, v in r.items() if k != "content"},
                }
                rec.update(
                    zip(("mem_active", "mem_cache", "mem_sum", "mem_peak"), mem)
                )
                fh.write(json.dumps(rec) + "\n")

    print()
    if failures:
        for line in failures:
            print(f"  FAIL  {line}")
        return 1
    print(f"  PASS  {len(sizes)} follow-up turn(s) admitted and served from the snapshot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
