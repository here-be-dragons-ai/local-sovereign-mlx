#!/usr/bin/env python3
"""Check kolibri1_1/ against a numpy forward of the llama.cpp #32891 semantics.

Tiny random models on the CPU in fp32; no checkpoint needed.

  nvfp4     round_nvfp4 against a search over all E4M3 / E2M1 values, with
            inputs on the rounding boundaries
  topk      topk_mask: ties go to the lowest index, -inf is never picked
  forward   logits of one pass against the numpy forward, which works row by
            row with explicit key sets; two layouts: pool = whole context,
            and pool = best blocks
  cache     chunked prefill plus token-by-token decode against one pass;
            both in fp32, but a value on a rounding boundary can still flip
            (seen once: one V element, 2e-4 in the hidden state)
  rows      routing scores computed in row chunks give the same mask

The numpy forward shares no code with the package. Two things keep the
comparison from failing on rounding noise instead of on errors:

- NVFP4 rounding is discrete, and among 10^5 values some lie within 1e-6 of a
  rounding boundary, where fp32 and fp64 round apart. The numpy forward
  therefore computes every tensor before rounding itself, reports how far it
  is from the package's, and then rounds the package's fp32 values with its
  own rounding. Without this, 3 of 8 seeds differ by up to 0.1 in a logit.
- It reports the smallest non-zero gap at a selection boundary. Exact ties
  (score 0 after the ReLU) are frequent and resolved by index on both sides.

    ~/src/mlx/.venv/bin/python verify-kolibri11.py [--seed N] [--tokens N]
"""

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N_SINK, N_RECENT, BLOCK = 8, 128, 16


def use_local(name):
    """Load ./<name>/ as mlx_vlm.models.<name> for this process."""
    full = f"mlx_vlm.models.{name}"
    import mlx_vlm.models  # noqa: F401

    for key in [k for k in sys.modules if k == full or k.startswith(full + ".")]:
        del sys.modules[key]
    pkg = HERE / name
    spec = importlib.util.spec_from_file_location(
        full, pkg / "__init__.py", submodule_search_locations=[str(pkg)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return importlib.import_module(full + ".language")


# ── numpy reference ──────────────────────────────────────────────────────────

E4M3 = np.array(
    [(c & 7) * 2.0**-9 if c < 8 else 2.0 ** ((c >> 3) - 7) * (1 + (c & 7) / 8) for c in range(1, 127)],
    dtype=np.float64,
)
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)


def nearest_even(x, grid, first_code):
    """Nearest grid value per element; a tie goes to the even code."""
    d = np.abs(x[..., None].astype(np.float64) - grid)
    lo = d.argmin(axis=-1)  # first of two tied neighbours
    hi = np.minimum(lo + 1, len(grid) - 1)
    tied = np.take_along_axis(d, hi[..., None], -1)[..., 0] == np.take_along_axis(d, lo[..., None], -1)[..., 0]
    pick = np.where(tied & ((lo + first_code) % 2 == 1), hi, lo)
    return grid[pick]


def np_nvfp4(x):
    x = np.asarray(x, dtype=np.float32)
    g = x.reshape(*x.shape[:-1], x.shape[-1] // 16, 16)
    amax = np.abs(g).max(axis=-1, keepdims=True)
    s = np.maximum(amax / np.float32(6.0), np.float32(2.0**-9))
    scale = nearest_even(s, E4M3, 1).astype(np.float32)
    y = g / scale
    r = nearest_even(np.abs(y), E2M1, 0).astype(np.float32)
    return (np.copysign(r, y) * scale).reshape(x.shape)


def rms(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def silu(x):
    return x / (1 + np.exp(-x))


def swiglu_mlp(x, W, p):
    return (silu(x @ W[p + ".gate_proj.weight"].T) * (x @ W[p + ".up_proj.weight"].T)) @ W[p + ".down_proj.weight"].T


def rope(x, pos, base):
    """mlx nn.RoPE, traditional=False: halves rotated against each other."""
    d = x.shape[-1] // 2
    theta = pos[:, None] * np.exp(-np.arange(d) * (np.log(base) / d))
    c, s = np.cos(theta), np.sin(theta)
    x1, x2 = x[..., :d], x[..., d:]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def softmax_attend(q, K, V, idx, scale):
    """q [H, D]; K, V [Hkv, N, D]; attend over the positions idx."""
    H, rep = q.shape[0], q.shape[0] // K.shape[0]
    out = []
    for h in range(H):
        g = h // rep
        s = (K[g, idx] @ q[h]) * scale
        a = np.exp(s - s.max())
        out.append((a / a.sum()) @ V[g, idx])
    return np.concatenate(out)


def best(scores, allowed, k):
    """The k best allowed indices, ties to the lowest index. Also the gap to the next one."""
    order = sorted(allowed, key=lambda i: (-scores[i], i))
    gap = np.inf
    if len(order) > k:
        top = max(abs(scores[order[0]]), 1e-30)
        gap = (scores[order[k - 1]] - scores[order[k]]) / top
    return order[:k], gap


class Rounder:
    """NVFP4 by np_nvfp4, applied to the package's value of each tensor."""

    def __init__(self, recorded):
        self.recorded, self.worst = iter(recorded), 0.0

    def __call__(self, x, perm=None):
        theirs = next(self.recorded)[0]
        if perm:
            theirs = theirs.transpose(perm)
        self.worst = max(self.worst, np.abs(theirs - x).max() / np.abs(x).max())
        return np_nvfp4(theirs).astype(np.float64)

    def done(self):
        return next(self.recorded, None) is None


def np_forward(W, cfg, tokens, rnd):
    """Logits [T, vocab] and the smallest relative gap at a selection boundary.

    rnd rounds in the order of the package: K, V, routing queries, block keys.
    """
    T = len(tokens)
    H, Hkv, D = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps, win = cfg["rms_norm_eps"], cfg["sliding_window"]
    R, topk = cfg["index_num_heads"], cfg["index_topk"]
    pool_blocks = cfg["index_pool_size"] // BLOCK
    scale = D**-0.5
    pos = np.arange(T, dtype=np.float64)
    h = W["model.embed_tokens.weight"][tokens]
    kv, pools, min_gap = {}, None, np.inf

    for i, kind in enumerate(cfg["layer_types"]):
        p = f"model.layers.{i}"
        a = rms(h, W[p + ".input_layernorm.weight"], eps)
        q = (a @ W[p + ".self_attn.q_proj.weight"].T).reshape(T, H, D)
        q = rms(q, W[p + ".self_attn.q_norm.weight"], eps)
        src = cfg["kv_source_layers"][i]
        if src == i:
            k = (a @ W[p + ".self_attn.k_proj.weight"].T).reshape(T, Hkv, D)
            k = rms(k, W[p + ".self_attn.k_norm.weight"], eps)
            v = (a @ W[p + ".self_attn.v_proj.weight"].T).reshape(T, Hkv, D)

        r = np.zeros((T, H * D))
        if kind == "sliding_attention":
            q = rope(q.transpose(1, 0, 2), pos, cfg["rope_theta"])  # [H, T, D]
            K = rope(k.transpose(1, 0, 2), pos, cfg["rope_theta"])
            V = v.transpose(1, 0, 2)
            for t in range(T):
                idx = np.arange(max(0, t - win + 1), t + 1)
                r[t] = softmax_attend(q[:, t], K, V, idx, scale)
        else:
            if src == i:
                kv[i] = (rnd(k, (1, 0, 2)).transpose(1, 0, 2), rnd(v, (1, 0, 2)).transpose(1, 0, 2))
            K, V = kv[src]
            k_route = K[0]  # [T, D]
            q_route = rnd(q[:, :R], (1, 0, 2))  # [T, R, D]
            indexer = i in cfg["index_layers"]
            if indexer:
                pools = [None] * T
                if T // BLOCK > pool_blocks:
                    nb = T // BLOCK
                    k_blocks = rnd(k_route[: nb * BLOCK].reshape(nb, BLOCK, D).mean(1))
            for t in range(T):
                attend = set(range(min(N_SINK, t + 1))) | set(range(max(0, t - N_RECENT + 1), t + 1))
                cand = list(range(N_SINK, t - N_RECENT + 1))
                if cand:
                    if indexer:
                        pools[t] = cand
                        nb = (t + 1) // BLOCK
                        if nb > pool_blocks:
                            bs = np.maximum(k_blocks[:nb] @ q_route[t].T, 0).mean(1) * scale
                            blocks, gap = best(bs, range(nb), pool_blocks)
                            min_gap = min(min_gap, gap if gap > 0 else np.inf)
                            inside = {b * BLOCK + j for b in blocks for j in range(BLOCK)}
                            pools[t] = [c for c in cand if c in inside]
                    s = np.maximum(k_route @ q_route[t].T, 0).mean(1) * scale
                    picked, gap = best(s, pools[t], topk)
                    min_gap = min(min_gap, gap if gap > 0 else np.inf)
                    attend |= set(picked)
                r[t] = softmax_attend(q[t], K, V, np.array(sorted(attend)), scale)

        r = r @ W[p + ".self_attn.o_proj.weight"].T
        h = h + rms(r, W[p + ".post_attn_norm.weight"], eps)
        x = rms(h, W[p + ".post_attention_layernorm.weight"], eps)
        if i in cfg["mlp_only_layers"]:
            r = swiglu_mlp(x, W, p + ".mlp")
        else:
            logits = x @ W[p + ".mlp.gate.weight"].T
            sel = np.argsort(-(logits + W[p + ".mlp.expert_bias"]), axis=-1, kind="stable")[:, : cfg["num_experts_per_tok"]]
            r = swiglu_mlp(x, W, p + ".mlp.shared_experts")
            for t in range(T):
                for e in sel[t]:
                    g = x[t] @ W[p + ".mlp.experts.gate_proj.weight"][e].T
                    u = x[t] @ W[p + ".mlp.experts.up_proj.weight"][e].T
                    y = (silu(g) * u) @ W[p + ".mlp.experts.down_proj.weight"][e].T
                    r[t] += y / (1 + np.exp(-logits[t, e]))
        h = h + rms(r, W[p + ".post_ffn_norm.weight"], eps)

    h = rms(h, W["model.norm.weight"], eps)
    return h @ W["lm_head.weight"].T, min_gap


# ── checks ───────────────────────────────────────────────────────────────────


def tiny_config(pool_size):
    types = ["sliding_attention", "sliding_attention", "indexed_attention", "sliding_attention",
             "indexed_attention", "sliding_attention", "sliding_attention", "indexed_attention",
             "sliding_attention", "indexed_attention"]
    return dict(
        model_type="kolibri1_1", layer_types=types, vocab_size=97, hidden_size=64,
        num_hidden_layers=10, num_attention_heads=8, num_key_value_heads=2, head_dim=16,
        rms_norm_eps=1e-6, rope_theta=10000.0, num_experts=6, num_experts_per_tok=2,
        moe_intermediate_size=32, shared_expert_intermediate_size=32, sliding_window=33,
        mlp_only_layers=[0, 1], mlp_only_intermediate_size=48,
        # 4 and 9 read the K/V of 2; 9 selects from the pool of indexer 7
        index_layers=[2, 7], kv_source_layers=[0, 1, 2, 3, 2, 5, 6, 7, 8, 2],
        index_topk=24, index_pool_size=pool_size, index_compression=BLOCK,
        index_num_local_tokens=N_RECENT, index_num_sink_tokens=N_SINK, index_num_heads=4,
    )


def build(lang, cfg, seed):
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    mx.random.seed(seed)
    model = lang.LanguageModel(lang.ModelConfig.from_dict(cfg))
    rng = np.random.default_rng(seed)
    params = dict(tree_flatten(model.parameters()))
    for k, v in params.items():
        if k.endswith("norm.weight") or k.endswith("layernorm.weight"):
            params[k] = mx.array(rng.uniform(0.5, 1.5, v.shape).astype(np.float32))
        elif k.endswith("expert_bias"):
            params[k] = mx.array(rng.normal(0, 0.5, v.shape).astype(np.float32))
        else:  # wider than the default init, so that the routing scores spread
            params[k] = mx.array(rng.normal(0, 2.0 / np.sqrt(v.shape[-1]), v.shape).astype(np.float32))
    model.update(tree_unflatten(list(params.items())))
    mx.eval(model.parameters())
    return model, {k: np.array(v).astype(np.float64) for k, v in params.items()}


def check_nvfp4(lang, rng):
    import mlx.core as mx

    x = rng.normal(0, 1, (4000, 16)).astype(np.float32) * np.exp(rng.uniform(-9, 7, (4000, 1))).astype(np.float32)
    # values on the E2M1 boundaries: the largest entry fixes the scale
    for row, s in zip(x[:1500], np.exp2(rng.integers(-6, 6, 1500)).astype(np.float32)):
        row[:] = rng.choice([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 0.2, 0.8, 4.9, 5.1], 16) * rng.choice([-1, 1], 16) * s
        row[0] = 6.0 * s
    # largest entry on the E4M3 boundaries of the scale, and beyond its range
    mids = (E4M3[:-1] + E4M3[1:]) / 2
    for row, m in zip(x[1500:2500], np.resize(mids, 1000)):
        row[0] = np.float32(m) * np.float32(6.0)
        row[1:] = np.clip(row[1:], -abs(row[0]), abs(row[0]))
    x[2500, 0], x[2501, :], x[2502, 0] = 6e3, 0.0, 1e-5
    got = np.array(lang.round_nvfp4(mx.array(x)))
    want = np_nvfp4(x)
    bad = int((got != want).sum())
    print(f"nvfp4    {x.size} values, {bad} differ")
    return bad == 0


def check_topk(lang, rng):
    import mlx.core as mx

    ok = True
    for _ in range(200):
        n, k = int(rng.integers(1, 40)), int(rng.integers(1, 12))
        s = rng.integers(0, 4, (3, n)).astype(np.float32)  # many ties
        s[rng.random((3, n)) < 0.3] = -np.inf
        got = np.array(lang.topk_mask(mx.array(s), k))
        for row, g in zip(s, got):
            live = [i for i in range(n) if row[i] > -np.inf]
            want = set(sorted(live, key=lambda i: (-row[i], i))[:k])
            ok &= set(np.flatnonzero(g)) == want
    print("topk     ties to the lowest index:", "ok" if ok else "MISMATCH")
    return ok


def check_model(lang, name, pool_size, seed, T):
    import mlx.core as mx

    cfg = tiny_config(pool_size)
    model, W = build(lang, cfg, seed)
    tokens = np.random.default_rng(seed + 1).integers(0, cfg["vocab_size"], T)
    x = mx.array(tokens)[None]

    recorded, package_round = [], lang.round_nvfp4

    def recording_round(a):
        recorded.append(np.array(a.astype(mx.float32)))
        return package_round(a)

    lang.round_nvfp4 = recording_round
    got = np.array(model(x).logits[0])
    lang.round_nvfp4 = package_round

    rnd = Rounder(recorded)
    want, gap = np_forward(W, cfg, tokens, rnd)
    scale = np.abs(want).max()
    err = np.abs(got - want).max() / scale
    ok = err < 2e-5 and rnd.worst < 2e-5 and rnd.done()
    print(f"forward  {name}: max |Δlogit| / max |logit| = {err:.1e}, before rounding {rnd.worst:.1e}, "
          f"smallest selection gap {gap:.1e}", "ok" if ok else "MISMATCH")

    # prefill in chunks, then token by token
    cache, parts, i = model.make_cache(), [], 0
    for step in [150, 37, 100] + [1] * 12 + [T]:
        if i >= T:
            break
        parts.append(model(x[:, i : i + step], cache=cache).logits[0])
        mx.eval(parts[-1], [c.state for c in cache])
        i += step
    err = np.abs(np.array(mx.concatenate(parts)) - got).max() / scale
    ok2 = err < 2e-5
    print(f"cache    {name}: chunked prefill + decode vs one pass = {err:.1e}", "ok" if ok2 else "MISMATCH")

    # routing scores in row chunks
    old, lang.SCORE_BYTES = lang.SCORE_BYTES, 4 * 4 * T * 7  # 7 rows at a time
    err = np.abs(np.array(model(x).logits[0]) - got).max() / scale
    lang.SCORE_BYTES = old
    ok3 = err == 0
    print(f"rows     {name}: scores in row chunks vs at once = {err:.1e}", "ok" if ok3 else "MISMATCH")
    return ok and ok2 and ok3


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=420)
    args = ap.parse_args()

    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    lang = use_local("kolibri1_1")
    rng = np.random.default_rng(args.seed)

    ok = check_nvfp4(lang, rng)
    ok &= check_topk(lang, rng)
    # 26 blocks at 420 tokens: 64 blocks hold them all, 4 do not
    ok &= check_model(lang, "pool = whole context", 64 * BLOCK, args.seed, args.tokens)
    ok &= check_model(lang, "pool = 4 best blocks", 4 * BLOCK, args.seed, args.tokens)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
