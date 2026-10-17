#!/usr/bin/env python3
"""Greedy-output fingerprint of a running server, for before/after an upgrade.

Sends one fixed request -- "X " + the first 6000 characters of the README at
commit 7a0e703, 250 tokens, temperature 0 -- and prints the first 12 hex
digits of sha256(reasoning + NUL + content). Same hash = same text, token for
token. Run it twice: the first request is cold, the second an exact-APC hit,
and both must agree.

    ./measure-greedy-hash.py --label before        # against 127.0.0.1:8888
    ./measure-greedy-hash.py --url http://127.0.0.1:8899 --label after --save /tmp/after.txt

Start the server with a FRESH STATE_DIR and ENABLE_SPEC_DECODE=0. Against a
live STATE_DIR the first request restores a snapshot from the SSD tier that an
older stack computed, so the prefill under test never runs (see
docs/upstream-2026-09-29.md). The hash is only comparable between runs of THIS
script: other prefixes or lengths give other texts.

A changed hash is not yet a defect. Find the first differing character with
--save, then look at the next-token logits there: on 2026-10-17 the texts
parted on an exact bf16 tie (docs/upstream-2026-10-17.md).
"""

import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

PROMPT_COMMIT = "7a0e703"
PROMPT_CHARS = 6000


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="Qwen3.8-27B-local")
    ap.add_argument("--label", default="run")
    ap.add_argument("--save", type=Path, help="write the full output here")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    readme = subprocess.run(
        ["git", "-C", str(here), "show", f"{PROMPT_COMMIT}:README.md"],
        check=True, capture_output=True, text=True,
    ).stdout
    body = {
        "model": args.model,
        "temperature": 0,
        "max_tokens": 250,
        "seed": 0,
        "messages": [{"role": "user", "content": "X " + readme[:PROMPT_CHARS]}],
    }
    req = urllib.request.Request(
        f"{args.url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = json.load(urllib.request.urlopen(req, timeout=900))
    msg = r["choices"][0]["message"]
    out = (msg.get("reasoning_content") or msg.get("reasoning") or "") + "\x00" + (msg.get("content") or "")
    usage = r.get("usage", {})
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    digest = hashlib.sha256(out.encode()).hexdigest()[:12]
    print(f"{args.label}: {digest}  prompt={usage.get('prompt_tokens')} cached={cached} out={usage.get('completion_tokens')}")
    if args.save:
        args.save.write_text(out)


if __name__ == "__main__":
    main()
