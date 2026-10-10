#!/usr/bin/env python3
"""Rename weight keys of an MLX checkpoint in place, e.g. for baseline builds.

Kolibri builds made with mlx-lm (velaia, eins78, tobiasoberrauch) store the
routed experts as `mlp.switch_mlp.*`; the kolibri1 model of mlx-vlm expects
`mlp.experts.*`. Same tensors, other names. This rewrites the shards, the
index and per-layer quantization entries in config.json, one shard at a time.

    ./rename-keys.py <dir> .mlp.switch_mlp. .mlp.experts.
"""

import json
import sys
from pathlib import Path

import mlx.core as mx


def main():
    d, old, new = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    for f in sorted(d.glob("*.safetensors")):
        arrays, meta = mx.load(str(f), return_metadata=True)
        if not any(old in k for k in arrays):
            continue
        arrays = {k.replace(old, new): v for k, v in arrays.items()}
        tmp = d / ("tmp-" + f.name)
        mx.save_safetensors(str(tmp), arrays, metadata=meta)
        del arrays
        tmp.replace(f)
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        j = json.loads(idx.read_text())
        j["weight_map"] = {k.replace(old, new): v for k, v in j["weight_map"].items()}
        idx.write_text(json.dumps(j, indent=2))
    cfg = d / "config.json"
    c = json.loads(cfg.read_text())
    for key in ("quantization", "quantization_config"):
        if isinstance(c.get(key), dict):
            c[key] = {k.replace(old, new): v for k, v in c[key].items()}
    cfg.write_text(json.dumps(c, indent=2))
    print(f"renamed {old} -> {new} in {d}")


if __name__ == "__main__":
    main()
