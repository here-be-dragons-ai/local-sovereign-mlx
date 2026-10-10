#!/usr/bin/env python3
"""Where does Kolibri's 3-bit build lose its accuracy? Error per layer and tensor group (sovereign-models#28).

One layer-streamed pass of the FP8 release over a few chat and Wikipedia
sequences. At each decoder layer the FP8 layer runs on its input, and so do
copies of the same layer with some tensors quantized; the difference of their
outputs is the error that quantization adds in this layer alone, on exactly
the input the original sees. Nothing propagates, so the layers can be
compared with each other.

Variants (bits, group size 64, affine as in convert-kolibri.py):
  shipped       experts 3, everything else 6, router bf16 (the current build)
  gate3/up3/down3   only that expert projection at 3, the rest unquantized
  attn6         only attention at 6
  shared6       only the shared expert at 6
  experts4      experts 4, everything else 6
  down4         like shipped, but expert down_proj at 4

Error per token: |out_variant - out_fp8| / |out_fp8 - in|, i.e. relative to
what the layer adds to the residual stream, reported as the ratio of sums
over all tokens of a group (chat, wiki). Also counted: how often each of the
384 experts is routed to, per layer and group.

    ./diagnose-kolibri-layers.py [--seqs 2] [--tokens 2048] [--variants shipped,down3,...]

Uses the kolibri profile of measure-quality.py (sets, FP8 path). Stop the
server first; the run streams the 73 GB release once.
"""

import argparse
import copy
import importlib.util
import json
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
GROUP = 64
VARIANTS = {
    "shipped": {"experts": 3, "other": 6},
    "gate3": {"experts.gate_proj": 3},
    "up3": {"experts.up_proj": 3},
    "down3": {"experts.down_proj": 3},
    "attn6": {"self_attn": 6},
    "shared6": {"shared_experts": 6},
    "experts4": {"experts": 4, "other": 6},
    "down4": {"experts": 3, "experts.down_proj": 4, "other": 6},
}


def _mq():
    spec = importlib.util.spec_from_file_location("_mq", _HERE / "measure-quality.py")
    mq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mq)
    mq.use_profile("kolibri")
    return mq


def bits_for(path, recipe):
    """Bits for a layer-relative module path under a variant, or None (unquantized)."""
    if path.endswith("mlp.gate"):
        return None
    best, best_len = None, -1
    for key, bits in recipe.items():
        if key != "other" and key in path and len(key) > best_len:
            best, best_len = bits, len(key)
    if best is not None:
        return best
    return recipe.get("other")


def quantized_copy(layer, recipe):
    import mlx.core as mx
    import mlx.nn as nn

    q = copy.deepcopy(layer)

    def pred(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        b = bits_for(path, recipe)
        return False if b is None else {"group_size": GROUP, "bits": b, "mode": "affine"}

    nn.quantize(q, class_predicate=pred)
    mx.eval(q.parameters())
    return q


def run_layer(layer, x, mask):
    """The layer's forward, also returning the routed expert indices."""
    import mlx.core as mx

    r = layer.self_attn(layer.input_layernorm(x), mask, None)
    h = x + layer.post_attn_norm(r)
    xm = layer.post_attention_layernorm(h)
    moe = layer.mlp
    logits = xm.astype(mx.float32) @ moe.gate.weight.astype(mx.float32).T
    k = moe.num_experts_per_tok
    inds = mx.argpartition(-(logits + moe.expert_bias), kth=k - 1, axis=-1)[..., :k]
    return h + layer.post_ffn_norm(moe(xm)), inds


def main():
    import mlx.core as mx
    from mlx_vlm.models.base import create_attention_mask

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seqs", type=int, default=2, help="sequences per group")
    ap.add_argument("--tokens", type=int, default=2048, help="tokens per sequence")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--layers", type=int, help="stop after this many layers (smoke test)")
    args = ap.parse_args()
    mq = _mq()
    data = Path(mq.DEFAULT_DATA).expanduser()
    sets = mq.load_sets(data, ["chat", "text"])
    seqs = [("chat", s[: args.tokens]) for s in sets["chat"]["seqs"][: args.seqs]]
    seqs += [("wiki", s[: args.tokens]) for s in sets["text"]["seqs"][: args.seqs]]
    variants = args.variants.split(",")
    out_dir = data / "diagnose"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"layers-{time.strftime('%Y-%m-%d')}.json"

    lm = mq.load_lazy(mq.ckpt_path(mq.REFERENCE))
    inner = lm.model
    hs = []
    for _, s in seqs:
        h = inner.embed_tokens(mx.array([s]))
        mx.eval(h)
        hs.append(h)
    inner.embed_tokens = None
    n_experts = inner.layers[0].mlp.gate.weight.shape[0]
    result = {"date": time.strftime("%Y-%m-%d"), "seqs": [(g, len(s)) for g, s in seqs],
              "variants": {v: VARIANTS[v] for v in variants}, "layers": []}
    n_layers = args.layers or len(inner.layers)
    for i in range(n_layers):
        t0 = time.time()
        layer = inner.layers[i]
        window = inner.sliding_window if layer.use_sliding else None
        mx.eval(layer.parameters())
        num = {(v, g): 0.0 for v in variants for g in ("chat", "wiki")}
        den = {g: 0.0 for g in ("chat", "wiki")}
        usage = {g: mx.zeros((n_experts,), dtype=mx.int32) for g in ("chat", "wiki")}
        masks = [create_attention_mask(h, None, window_size=window) for h in hs]
        outs = []
        for (g, _), h, m in zip(seqs, hs, masks):
            o, inds = run_layer(layer, h, m)
            usage[g] = usage[g] + mx.zeros((n_experts,), dtype=mx.int32).at[inds.flatten()].add(1)
            upd = (o - h).astype(mx.float32)
            den[g] += mx.sum(upd * upd).item()
            outs.append(o)
            mx.eval(o)
        for v in variants:
            q = quantized_copy(layer, VARIANTS[v])
            for (g, _), h, m, o in zip(seqs, hs, masks, outs):
                d = (q(h, m) - o).astype(mx.float32)
                num[(v, g)] += mx.sum(d * d).item()
            del q
            mx.clear_cache()
        entry = {"layer": i, "sliding": bool(layer.use_sliding),
                 "rel_error": {v: {g: (num[(v, g)] / den[g]) ** 0.5 for g in den} for v in variants},
                 "usage": {g: usage[g].tolist() for g in usage}}
        result["layers"].append(entry)
        hs = outs
        inner.layers[i] = None
        del layer
        mx.clear_cache()
        out.write_text(json.dumps(result))
        s = entry["rel_error"].get("shipped")
        print(f"[{time.strftime('%H:%M:%S')}] layer {i:2d} {'swa' if entry['sliding'] else 'FULL'}  "
              + (f"shipped chat {s['chat']:.4f} wiki {s['wiki']:.4f}  " if s else "")
              + f"{time.time() - t0:5.1f} s  peak {mx.get_peak_memory() / 2**30:.1f} GiB", flush=True)
    print(f"raw: {out}")


if __name__ == "__main__":
    main()
