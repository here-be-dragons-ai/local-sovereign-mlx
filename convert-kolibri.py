#!/usr/bin/env python3
"""Convert Aleph-Alpha/Kolibri-1 (block-FP8) to a mixed 3/6-bit MLX checkpoint.

Needs the kolibri1 model package in mlx_vlm (patch 0050).

Recipe, sized for 48 GB with iogpu.wired_limit_mb=40960:
  routed experts       3 bit, group 64   (75.5B of 78.1B parameters)
  everything else      6 bit, group 64   (attention, shared expert,
                                           embedding, lm_head)
  router (mlp.gate)    bf16, as in the release; expert_bias stays fp32
That gives ~35 GB of weights. All-4-bit would be ~44 GB and does not fit.

The FP8 weights are dequantized to bf16 in Kolibri1's sanitize and quantized
once. mlx-vlm's generic FP8 loader would first requantize them to MXFP8, so
the conversion reads them through a staging directory whose config.json has
the checkpoint's quantization_config removed.

Usage:
  ./convert-kolibri.py ~/src/mlx/models/Kolibri-1-FP8 ~/src/mlx/models/Kolibri-1-MLX-3bit

--other-bits 3 builds the uniform 3-bit control for the quality ablation in
measure-kolibri-quality.py (issue #8); it is not a recipe to ship.
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

EXPERT_BITS = 3
OTHER_BITS = 6
GROUP_SIZE = 64


def make_predicate(other_bits):
    def predicate(path, module):
        if path.endswith("mlp.gate"):
            return False
        if ".mlp.experts." in path:
            return {"group_size": GROUP_SIZE, "bits": EXPERT_BITS, "mode": "affine"}
        return {"group_size": GROUP_SIZE, "bits": other_bits, "mode": "affine"}

    return predicate


def make_staging(src: Path) -> Path:
    staging = Path(tempfile.mkdtemp(prefix="kolibri-staging-"))
    for f in src.iterdir():
        if f.name != "config.json":
            (staging / f.name).symlink_to(f.resolve())
    config = json.loads((src / "config.json").read_text())
    if config.pop("quantization_config", None) is None:
        print("[WARN] source config has no quantization_config; not the FP8 release?")
    (staging / "config.json").write_text(json.dumps(config, indent=2))
    return staging


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--other-bits", type=int, default=OTHER_BITS,
                    help="bits outside the routed experts (default %(default)s)")
    args = ap.parse_args()

    src, dst = args.src.expanduser(), args.dst.expanduser()
    shards = sorted(src.glob("model-*.safetensors"))
    index = json.loads((src / "model.safetensors.index.json").read_text())
    expected = set(index["weight_map"].values())
    missing = expected - {s.name for s in shards}
    if missing:
        sys.exit(f"[ERROR] {len(missing)} shard(s) missing, download incomplete")
    if dst.exists():
        sys.exit(f"[ERROR] {dst} exists")

    from mlx_vlm.convert import convert

    staging = make_staging(src)
    try:
        convert(
            str(staging),
            mlx_path=str(dst),
            quantize=True,
            q_group_size=GROUP_SIZE,
            q_bits=EXPERT_BITS,
            quant_predicate=make_predicate(args.other_bits),
        )
    finally:
        shutil.rmtree(staging)
    print(f"[INFO] done: {dst}")


if __name__ == "__main__":
    main()
