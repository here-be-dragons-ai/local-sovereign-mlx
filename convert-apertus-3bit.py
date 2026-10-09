#!/usr/bin/env python3
"""Convert swiss-ai Apertus 1.5 (bf16) to a text-only 3/6-bit MLX checkpoint.

Needs the apertus1p5 model package (mlx-vlm branch apertus1p5); run it with
~/src/mlx/.venv-apertus/bin/python.

Recipe, sized for Apertus-v1.5-70B on 48 GB with iogpu.wired_limit_mb=40960:
  decoder linears           3 bit affine, group 64
  embed_tokens, lm_head     6 bit affine, group 64
  image / audio tokenizer   dropped (text only)
That gives ~30.4 GiB of weights for the 70B, leaving ~20k tokens of f16 KV
cache (320 KiB/token). All-4-bit would be ~38 GiB and does not fit.

The tokenizers are dropped through a staging directory: its config.json has
vision_tokenizer_config / audio_tokenizer_config removed, so the model does not
build those modules and its sanitize discards their weights, and the
tokenizer shards are not linked at all.

Plain round-to-nearest 3 bit breaks Apertus (up_proj in front of xIELU), so
the published 70B build applies AWQ scales first: awq-apertus-scales.py
computes them layer by layer, --awq-scales folds them into the norms before
quantization. (mlx-vlm's own AWQ runs a full bf16 forward pass, which would
hold the 140 GB bf16 model in memory, and skips Apertus.)

Usage, as for here-be-dragons-ai/Apertus-v1.5-70B-MLX-3bit:
  awq-apertus-scales.py   ~/src/mlx/models/Apertus-v1.5-70B scales-70b.npz
  convert-apertus-3bit.py ~/src/mlx/models/Apertus-v1.5-70B ~/src/mlx/models/Apertus-v1.5-70B-MLX-3bit \
      --awq-scales scales-70b.npz
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

LOW_BITS = 3
HIGH_BITS = 6
UP_BITS = None  # up_proj override (None = LOW_BITS)
DOWN_BITS = None  # down_proj override
GROUP_SIZE = 64
TOKENIZER_SHARDS = ("model-vision_tokenizer", "model-wavtokenizer")


def predicate(path, module):
    from mlx_vlm.utils import skip_multimodal_module

    if skip_multimodal_module(path) or not hasattr(module, "to_quantized"):
        return False
    if module.weight.shape[-1] % GROUP_SIZE != 0:
        return False
    if path.endswith(("embed_tokens", "lm_head")):
        bits = HIGH_BITS
    elif path.endswith("up_proj") and UP_BITS:
        bits = UP_BITS
    elif path.endswith("down_proj") and DOWN_BITS:
        bits = DOWN_BITS
    else:
        bits = LOW_BITS
    return {"group_size": GROUP_SIZE, "bits": bits, "mode": "affine"}


def mx_load(path: Path):
    import mlx.core as mx

    return mx.load(str(path))


def apply_awq_scales(model, scales) -> int:
    """Fold AWQ scales (awq-apertus-scales.py) into the lazily loaded weights;
    the bf16 function is unchanged up to rounding."""
    layers = model.language_model.model.layers
    used = 0

    def fold(norm_or_prev, linears, s, into_rows=False):
        inv = 1.0 / s
        w = norm_or_prev.weight
        norm_or_prev.weight = (w * (inv[:, None] if into_rows else inv)).astype(w.dtype)
        for lin in linears:
            lin.weight = (lin.weight * s).astype(lin.weight.dtype)

    for i, layer in enumerate(layers):
        a, mlp = layer.self_attn, layer.mlp
        if (s := scales.get(f"layers.{i}.qkv")) is not None:
            fold(layer.attention_layernorm, [a.q_proj, a.k_proj, a.v_proj], s)
            used += 1
        if (s := scales.get(f"layers.{i}.up")) is not None:
            fold(layer.feedforward_layernorm, [mlp.up_proj], s)
            used += 1
        if (s := scales.get(f"layers.{i}.o")) is not None:
            fold(a.v_proj, [a.o_proj], s, into_rows=True)
            used += 1
    if used != len(scales):
        raise ValueError(f"used {used} of {len(scales)} scale vectors")
    return used


def make_staging(src: Path) -> Path:
    staging = Path(tempfile.mkdtemp(prefix="apertus-staging-"))
    for f in src.iterdir():
        if f.name in ("config.json", "model.safetensors.index.json"):
            continue
        if f.name.startswith(TOKENIZER_SHARDS) or f.name.startswith("."):
            continue
        (staging / f.name).symlink_to(f.resolve())

    config = json.loads((src / "config.json").read_text())
    for key in ("vision_tokenizer_config", "audio_tokenizer_config"):
        if config.pop(key, None) is None:
            print(f"[WARN] source config has no {key}")
    (staging / "config.json").write_text(json.dumps(config, indent=2))

    index = json.loads((src / "model.safetensors.index.json").read_text())
    index["weight_map"] = {
        k: v
        for k, v in index["weight_map"].items()
        if not v.startswith(TOKENIZER_SHARDS)
    }
    (staging / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    return staging


def main():
    global LOW_BITS, UP_BITS, DOWN_BITS, GROUP_SIZE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--awq-scales", type=Path, help="npz from awq-apertus-scales.py")
    ap.add_argument("--bits", type=int, default=LOW_BITS, help="decoder linears")
    ap.add_argument("--up-bits", type=int, help="up_proj, if different from --bits")
    ap.add_argument("--down-bits", type=int, help="down_proj, if different from --bits")
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    args = ap.parse_args()
    LOW_BITS, UP_BITS, DOWN_BITS = args.bits, args.up_bits, args.down_bits
    GROUP_SIZE = args.group_size

    src, dst = args.src.expanduser(), args.dst.expanduser()
    if not (src / "config.json").exists():
        sys.exit(f"no checkpoint at {src}")
    if dst.exists():
        sys.exit(f"{dst} exists; remove it first")

    from mlx_vlm.convert import convert

    # mlx_vlm.convert as an attribute is the function; patch the module.
    mvc = sys.modules["mlx_vlm.convert"]

    if args.awq_scales:
        scales = mx_load(args.awq_scales.expanduser())
        fetch = mvc.fetch_from_hub

        def fetch_scaled(*a, **kw):
            model, config, processor = fetch(*a, **kw)
            print(f"[INFO] AWQ: folding {apply_awq_scales(model, scales)} scale vectors")
            return model, config, processor

        mvc.fetch_from_hub = fetch_scaled

    staging = make_staging(src)
    try:
        convert(
            str(staging),
            mlx_path=str(dst),
            quantize=True,
            q_group_size=GROUP_SIZE,
            q_bits=LOW_BITS,
            dtype="bfloat16",
            quant_predicate=predicate,
        )
    finally:
        shutil.rmtree(staging)

    config = json.loads((dst / "config.json").read_text())
    leftover = [k for k in ("vision_tokenizer_config", "audio_tokenizer_config") if k in config]
    if leftover:
        print(f"[WARN] output config still has {leftover}")
    print(f"[INFO] done: {dst}")


if __name__ == "__main__":
    main()
