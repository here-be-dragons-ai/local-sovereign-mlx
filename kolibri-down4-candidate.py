#!/usr/bin/env python3
"""Kolibri candidate: the shipped 3/6-bit build with expert down_proj at 4 bit in chosen layers (sovereign-models#28).

The per-layer diagnosis (diagnose-kolibri-layers.py) puts most of the
quantization error into the routed experts' down_proj, concentrated in a dozen
layers. This builds a test candidate without rewriting 33 GB: the shipped
build is cloned (APFS copy-on-write, no extra space), the new down_proj
tensors are quantized from the FP8 release into one extra shard, and the index
and config point to it. mlx-vlm loads the shards in sorted order, so the extra
shard (model-zz-*) overrides the 3-bit tensors left in the cloned shards.
Fine for measuring; a build to publish would be written cleanly with
convert-kolibri.py.

    ./kolibri-down4-candidate.py --layers 29,16,35,30,41,48,38,27,31,42,32,36,43 \
        --out ~/src/mlx/models/Kolibri-1-MLX-3bit-down4
"""

import argparse
import importlib.util
import json
import shutil
import subprocess
import time
from pathlib import Path

MODELS = Path("~/src/mlx/models").expanduser()
SHARD = "model-zz-down4.safetensors"
GROUP = 64


def main():
    import mlx.core as mx

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--layers", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--base", type=Path, default=MODELS / "Kolibri-1-MLX-3bit")
    ap.add_argument("--fp8", type=Path, default=MODELS / "Kolibri-1-FP8")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")

    subprocess.run(["cp", "-cR", str(args.base), str(args.out)], check=True)

    spec = importlib.util.spec_from_file_location("_mq", Path(__file__).resolve().parent / "measure-quality.py")
    mq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mq)
    lm = mq.load_lazy(args.fp8)

    index_path = args.out / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    ref_dtype = None
    out = {}
    for i in layers:
        t0 = time.time()
        prefix = f"model.layers.{i}.mlp.experts.down_proj"
        if ref_dtype is None:
            shard = index["weight_map"][f"{prefix}.scales"]
            ref_dtype = mx.load(str(args.out / shard))[f"{prefix}.scales"].dtype
        w = lm.model.layers[i].mlp.experts.down_proj.weight
        wq, scales, biases = mx.quantize(w, group_size=GROUP, bits=args.bits)
        out[f"{prefix}.weight"] = wq
        out[f"{prefix}.scales"] = scales.astype(ref_dtype)
        out[f"{prefix}.biases"] = biases.astype(ref_dtype)
        mx.eval(out[f"{prefix}.weight"], out[f"{prefix}.scales"], out[f"{prefix}.biases"])
        lm.model.layers[i] = None
        mx.clear_cache()
        print(f"layer {i}: down_proj {tuple(w.shape)} -> {args.bits} bit, {time.time() - t0:.0f} s", flush=True)
    mx.save_safetensors(str(args.out / SHARD), out, metadata={"format": "mlx"})

    for k in out:
        index["weight_map"][k] = SHARD
    tmp = index_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, indent=2))
    tmp.replace(index_path)

    config_path = args.out / "config.json"
    config = json.loads(config_path.read_text())
    for key in ("quantization", "quantization_config"):
        q = config.get(key)
        if isinstance(q, dict):
            for i in layers:
                q[f"model.layers.{i}.mlp.experts.down_proj"] = {"group_size": GROUP, "bits": args.bits, "mode": "affine"}
    tmp = config_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(config, indent=2))
    tmp.replace(config_path)
    size = (args.out / SHARD).stat().st_size
    print(f"{args.out}: {len(layers)} layers at {args.bits} bit, extra shard {size / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
