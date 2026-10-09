#!/usr/bin/env python3
"""Check a freshly converted checkpoint against a published build.

    ./verify-build.py <local dir> <hf repo> [--revision main]

File level: the SHA-256 of every local file against the Hub (the LFS object
id is the file's SHA-256; small files are downloaded and hashed). Where a
.safetensors file differs, tensor level: every tensor's name, dtype, shape
and a SHA-256 of its bytes, read from both files' headers and data, so
"same weights, different metadata or order" can be told apart from "different
weights". Remote tensors are read with HTTP range requests.

Exit code 0 when every file is identical or every tensor matches.
"""

import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path


def _token():
    path = Path(os.path.expanduser("~/.cache/huggingface/token"))
    return path.read_text().strip() if path.exists() else os.environ.get("HF_TOKEN")


TOKEN = _token()


def _get(url, rng=None):
    h = {"User-Agent": "local-sovereign-mlx/verify-build"}
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}"
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=300) as r:
        return r.read()


def sha256_file(path, chunk=64 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def remote_files(repo, revision):
    rev = json.loads(_get(f"https://huggingface.co/api/models/{repo}/revision/{revision}"))["sha"]
    tree = json.loads(_get(f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"))
    out = {}
    for e in tree:
        if e["type"] != "file" or e["path"] in (".gitattributes", "README.md"):
            continue
        lfs = e.get("lfs")
        out[e["path"]] = lfs["oid"] if lfs else hashlib.sha256(
            _get(f"https://huggingface.co/{repo}/resolve/{rev}/{e['path']}")).hexdigest()
    return rev, out


def tensors_local(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            a, b = info["data_offsets"]
            f.seek(base + a)
            out[name] = (info["dtype"], tuple(info["shape"]), hashlib.sha256(f.read(b - a)).hexdigest())
    return out, header.get("__metadata__", {})


def tensors_remote(url):
    n = struct.unpack("<Q", _get(url, (0, 7)))[0]
    header = json.loads(_get(url, (8, 8 + n - 1)))
    base = 8 + n
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        data = _get(url, (base + a, base + b - 1)) if b > a else b""
        out[name] = (info["dtype"], tuple(info["shape"]), hashlib.sha256(data).hexdigest())
    return out, header.get("__metadata__", {})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("local", type=Path)
    ap.add_argument("repo")
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()

    rev, remote = remote_files(args.repo, args.revision)
    local = {str(p.relative_to(args.local)): p for p in sorted(args.local.rglob("*"))
             if p.is_file() and p.name not in (".gitattributes", "README.md")
             and ".cache" not in p.parts}
    print(f"{args.repo} @ {rev}")
    ok = True
    for name in sorted(set(remote) | set(local)):
        if name not in local or name not in remote:
            print(f"  only {'remote' if name in remote else 'local'}: {name}")
            ok = False
            continue
        if sha256_file(local[name]) == remote[name]:
            print(f"  identical  {name}")
            continue
        if not name.endswith(".safetensors"):
            print(f"  DIFFERS    {name}")
            ok = False
            continue
        lt, lm = tensors_local(local[name])
        rt, rm = tensors_remote(f"https://huggingface.co/{args.repo}/resolve/{rev}/{name}")
        same = [k for k in lt if rt.get(k) == lt[k]]
        if len(same) == len(lt) == len(rt):
            print(f"  tensors identical, file bytes differ  {name}  (metadata {lm} vs {rm})")
        else:
            ok = False
            print(f"  DIFFERS    {name}: {len(same)} of {len(rt)} tensors identical; "
                  f"only local {len(set(lt) - set(rt))}, only remote {len(set(rt) - set(lt))}")
    print("RESULT", "match" if ok else "mismatch")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
