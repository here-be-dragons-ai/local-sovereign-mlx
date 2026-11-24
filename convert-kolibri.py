#!/usr/bin/env python3
"""Convert Aleph-Alpha/Kolibri-1 or Kolibri-1.1 (block-FP8) to a mixed 3/6-bit MLX checkpoint.

Runs on this repository's own packages (kolibri1/, kolibri1_1/): use_local()
puts them in place of mlx_vlm.models.<model_type> for this process. mlx-vlm
ships kolibri1 since #2423, but only the local package dequantizes the FP8
release, and its parameter names (mlp.experts.*, mlp.expert_bias) are the
ones the shipped checkpoint and the quality tools are written against.
mlx-vlm has no kolibri1_1.

Recipe, sized for 48 GB with iogpu.wired_limit_mb=40960:
  routed experts       3 bit, group 64   (75.5B of 78.1B parameters;
                                           1.1: 72.5B of 75.1B)
  everything else      6 bit, group 64   (attention, shared expert,
                                           embedding, lm_head; 1.1: also the
                                           dense MLP of layers 0 and 1)
  router (mlp.gate)    bf16, as in the release; expert_bias stays fp32
That gives ~35 GB of weights (1.1: ~34 GB). All-4-bit would be ~44 GB
(1.1: ~42 GB) and does not fit.

The FP8 weights are dequantized to bf16 in Kolibri1's sanitize and quantized
once. mlx-vlm's generic FP8 loader would first requantize them to MXFP8, so
the conversion reads them through a staging directory whose config.json has
the checkpoint's quantization_config removed.

Kolibri 1.1 lists its sparse layers as "indexed_attention". transformers
5.15 rejects that name when it reads config.json for the tokenizer, so the
staging copy and the converted build say "full_attention", as early 1.1
checkpoints did; kolibri1_1/ takes both.

The Kolibri 1.1 build carries its model code: kolibri1_1.py, the package as
one module, named by "model_file" in config.json. mlx-vlm (0.8.0) loads the
model from there, so the build runs on a stock server. After a change to
kolibri1_1/, --code-only writes it into an existing build again.

Usage:
  ./convert-kolibri.py ~/src/mlx/models/Kolibri-1-FP8 ~/src/mlx/models/Kolibri-1-MLX-3bit
  ./convert-kolibri.py ~/src/mlx/models/Kolibri-1.1-FP8 ~/src/mlx/models/Kolibri-1.1-MLX-3bit

--other-bits 3 builds the uniform 3-bit control for the quality ablation in
measure-kolibri-quality.py (issue #8); it is not a recipe to ship.
"""

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL = ("kolibri1", "kolibri1_1")

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


def use_local(model_type="kolibri1"):
    """Load ./<model_type>/ as mlx_vlm.models.<model_type> for this process.

    mlx-vlm resolves a model_type through importlib.import_module, which takes
    sys.modules first. kolibri1 was patch 0050 until upstream shipped its own
    (#2423), which loads the MLX checkpoints but not the FP8 release. Call it
    before anything loads a Kolibri model; a package already imported from
    mlx-vlm is replaced.
    """
    if model_type not in LOCAL:
        return
    local = HERE / model_type
    name = f"mlx_vlm.models.{model_type}"
    current = sys.modules.get(name)
    if current is not None and Path(current.__file__).parent == local:
        return
    import mlx_vlm.models  # noqa: F401  (the parent package must exist)

    for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        del sys.modules[key]
    spec = importlib.util.spec_from_file_location(
        name, local / "__init__.py", submodule_search_locations=[str(local)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def use_local_kolibri1():
    use_local("kolibri1")


def model_file_source(model_type: str) -> str:
    """The local package as one module with absolute imports."""
    out = [f"# {model_type}/ of local-sovereign-mlx as one module, written by convert-kolibri.py."]
    for name in ("config.py", "language.py", f"{model_type}.py"):
        for line in (HERE / model_type / name).read_text().splitlines():
            if line.startswith("from .."):
                line = "from mlx_vlm.models." + line[len("from ..") :]
            elif line.startswith("from ."):
                continue  # within the package
            out.append(line)
    return "\n".join(out) + "\n"


def make_staging(src: Path, fp8: bool = True, model_file: bool = False) -> Path:
    """A copy of src by symlinks whose config.json mlx-vlm and transformers take.

    fp8: drop quantization_config, so that the model's sanitize dequantizes.
    model_file: add the model code and name it in config.json.
    """
    staging = Path(tempfile.mkdtemp(prefix="kolibri-staging-"))
    for f in src.iterdir():
        if f.name != "config.json":
            (staging / f.name).symlink_to(f.resolve())
    config = json.loads((src / "config.json").read_text())
    if fp8 and config.pop("quantization_config", None) is None:
        print("[WARN] source config has no quantization_config; not the FP8 release?")
    if "layer_types" in config:
        config["layer_types"] = [
            "full_attention" if t == "indexed_attention" else t for t in config["layer_types"]
        ]
    if model_file:
        name = config["model_type"] + ".py"
        (staging / name).write_text(model_file_source(config["model_type"]))
        config["model_file"] = name
    (staging / "config.json").write_text(json.dumps(config, indent=2))
    return staging


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--other-bits", type=int, default=OTHER_BITS,
                    help="bits outside the routed experts (default %(default)s)")
    ap.add_argument("--code-only", action="store_true",
                    help="rewrite the model code in the existing build dst, convert nothing")
    args = ap.parse_args()

    src, dst = args.src.expanduser(), args.dst.expanduser()
    if args.code_only:
        name = json.loads((dst / "config.json").read_text()).get("model_file")
        if not name:
            sys.exit(f"[ERROR] {dst} names no model_file")
        (dst / name).write_text(model_file_source(Path(name).stem))
        print(f"[INFO] wrote {dst / name}")
        return
    shards = sorted(src.glob("model-*.safetensors"))
    index = json.loads((src / "model.safetensors.index.json").read_text())
    expected = set(index["weight_map"].values())
    missing = expected - {s.name for s in shards}
    if missing:
        sys.exit(f"[ERROR] {len(missing)} shard(s) missing, download incomplete")
    if dst.exists():
        sys.exit(f"[ERROR] {dst} exists")

    model_type = json.loads((src / "config.json").read_text()).get("model_type")
    use_local(model_type)
    from mlx_vlm.convert import convert

    # mlx-vlm ships kolibri1; a kolibri1_1 build has to bring its code.
    staging = make_staging(src, model_file=model_type == "kolibri1_1")
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
