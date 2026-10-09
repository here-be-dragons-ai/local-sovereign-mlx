#!/usr/bin/env python3
"""Rewrite an MLX checkpoint without the "language_model." key prefix.

convert-kolibri.py writes mlx-vlm's layout (language_model.*); the published
Kolibri-1-MLX-3bit uses the prefix-free layout that loads in both mlx-lm and
mlx-vlm. This rewrites every shard with the renamed keys (same shards, same
tensor order, same metadata) and fixes the index.

    ./strip-prefix.py <src dir> <dst dir>
    ./strip-prefix.py <dir>                 # in place, one shard at a time
"""

import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx

PREFIX = "language_model."


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if f.suffix == ".safetensors":
            arrays, meta = mx.load(str(f), return_metadata=True)
            arrays = {k.removeprefix(PREFIX): v for k, v in arrays.items()}
            tmp = dst / ("tmp-" + f.name)
            mx.save_safetensors(str(tmp), arrays, metadata=meta)
            del arrays
            tmp.replace(dst / f.name)
        elif f.name == "model.safetensors.index.json":
            idx = json.loads(f.read_text())
            idx["weight_map"] = {k.removeprefix(PREFIX): v for k, v in idx["weight_map"].items()}
            (dst / f.name).write_text(json.dumps(idx, indent=2))
        elif f.is_file() and dst != src:
            shutil.copy2(f, dst / f.name)
    print(f"done: {dst}")


if __name__ == "__main__":
    main()
