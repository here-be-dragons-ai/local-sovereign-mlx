#!/usr/bin/env python3
"""AWQ scales for Apertus 1.5, computed layer by layer so that the bf16 model
never has to fit in memory (the 70B is 140 GB; one layer is 1.7 GB).

The calibration hidden states are pushed through one bf16 decoder layer at a
time; the inputs of that layer's linears give the AWQ statistics, the scales
are searched against BITS-bit fake quantization, and the layer is dropped
before the next one is read from disk. Output: one .npz with the scale
vectors, applied by convert-apertus-3bit.py --awq-scales.

Folds (mlx-vlm's AWQ assumes gate_proj and skips Apertus):
  attention_layernorm   -> q_proj, k_proj, v_proj   key layers.N.qkv
  feedforward_layernorm -> up_proj                  key layers.N.up
  v_proj -> o_proj (only without GQA)               key layers.N.o
down_proj gets none: xIELU between up_proj and down_proj is nonlinear.

  awq-apertus-scales.py ~/src/mlx/models/Apertus-v1.5-70B scales-70b.npz
"""

import argparse
import importlib.util
import shutil
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.models.base import create_attention_mask
from mlx_vlm.quant.awq import _search_scale
from mlx_vlm.quant.calibration import collect_activation_stats

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("conv", HERE / "convert-apertus-3bit.py")
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)


def calibration_texts(src: Path):
    # The evaluation scripts use README[3000:9000]; calibrate elsewhere.
    readme = (src / "README.md").read_text()
    return [
        readme[12000:18000],
        "Der Föderalismus prägt die politische Kultur der Schweiz: Gemeinden erheben "
        "eigene Steuern, Kantone regeln Bildung und Polizei, der Bund Aussenpolitik "
        "und Landesverteidigung. Volksinitiativen und Referenden geben den "
        "Stimmberechtigten direkten Einfluss auf die Gesetzgebung. " * 3,
        "Le Conseil fédéral est composé de sept membres élus par l'Assemblée fédérale. "
        "Il siège à Berne et dirige l'administration fédérale selon le principe de "
        "collégialité. Les décisions sont prises en commun et défendues par tous. " * 3,
        "Il Ticino è l'unico cantone interamente di lingua italiana. Lugano è la sua "
        "città più grande, Bellinzona il capoluogo. " * 4,
        "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n"
        "    return a\n\nclass Stack:\n    def __init__(self):\n        self.items = []\n"
        "    def push(self, x):\n        self.items.append(x)\n    def pop(self):\n"
        "        return self.items.pop()\n" * 2,
        "<|system_start|>You are a helpful assistant.<|system_end|><|user_start|>Explain "
        "in two sentences why the sky is blue.<|user_end|><|assistant_start|>Sunlight "
        "is scattered by the molecules of the air, and short blue wavelengths scatter "
        "much more strongly than red ones. That scattered blue light reaches our eyes "
        "from every direction of the sky.<|assistant_end|>",
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--bits", type=int, default=conv.LOW_BITS)
    ap.add_argument("--group-size", type=int, default=conv.GROUP_SIZE)
    ap.add_argument("--up-bits", type=int, help="bits for up_proj, if different")
    ap.add_argument("--rows", type=int, default=512, help="input rows per linear for the search")
    ap.add_argument("--grid", type=int, default=20)
    args = ap.parse_args()
    src = args.src.expanduser()

    staging = conv.make_staging(src)
    try:
        model, processor = load(str(staging), lazy=True)
    finally:
        shutil.rmtree(staging)
    tok = processor.tokenizer
    lm = model.language_model
    layers = lm.model.layers

    seqs = [mx.array([tok.encode(t)]) for t in calibration_texts(src)]
    print(f"[INFO] {len(layers)} layers, {sum(s.shape[1] for s in seqs)} calibration tokens")

    # Resume: after every layer the scales so far and the hidden states that
    # enter the next layer are written to <out>.partial.npz.
    out = args.out.expanduser()
    partial = out.with_name(out.stem + ".partial.npz")
    scales, start = {}, 0
    if partial.exists():
        saved = mx.load(str(partial))
        start = int(saved.pop("_next_layer").item())
        hs = [saved.pop(f"_h{j}") for j in range(len(seqs))]
        scales = saved
        print(f"[INFO] resuming at layer {start}")
    else:
        hs = [lm.model.embed_tokens(s) for s in seqs]
        mx.eval(hs)
    lm.model.embed_tokens = None  # 4.4 GB on the 70B, no longer needed

    for i in range(start, len(layers)):
        t0 = time.time()
        layer = layers[i]
        outs = []

        def run():
            outs.clear()
            for h in hs:
                y = layer(h, mask=create_attention_mask(h, None))
                mx.eval(y)
                outs.append(y)

        stats = collect_activation_stats(layer, run, max_rows=args.rows)
        a, mlp = layer.self_attn, layer.mlp
        G, B, N = args.group_size, args.bits, args.grid
        st = stats["self_attn.q_proj"]
        scales[f"layers.{i}.qkv"] = _search_scale(
            [a.q_proj.weight, a.k_proj.weight, a.v_proj.weight],
            st["inputs"], st["scale"], G, B, N)
        st = stats["mlp.up_proj"]
        scales[f"layers.{i}.up"] = _search_scale(
            [mlp.up_proj.weight], st["inputs"], st["scale"], G, args.up_bits or B, N)
        if a.v_proj.weight.shape[0] == a.o_proj.weight.shape[-1]:
            st = stats["self_attn.o_proj"]
            scales[f"layers.{i}.o"] = _search_scale(
                [a.o_proj.weight], st["inputs"], st["scale"], G, B, N)
        mx.eval(list(scales.values()))

        hs = list(outs)
        del stats, layer, a, mlp, outs
        layers[i] = None  # drop the bf16 weights of this layer
        mx.clear_cache()
        print(f"[INFO] layer {i:3d}  {time.time() - t0:5.1f} s  "
              f"active {mx.get_active_memory() / 2**30:5.1f} GiB  "
              f"peak {mx.get_peak_memory() / 2**30:5.1f} GiB", flush=True)
        state = dict(scales, _next_layer=mx.array(i + 1))
        state.update({f"_h{j}": h for j, h in enumerate(hs)})
        mx.savez(str(partial.with_name("tmp-" + partial.name)), **state)
        partial.with_name("tmp-" + partial.name).replace(partial)

    mx.savez(str(out), **scales)
    partial.unlink(missing_ok=True)
    print(f"[INFO] {len(scales)} scale vectors -> {out}")


if __name__ == "__main__":
    main()
