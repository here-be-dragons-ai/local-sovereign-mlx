"""Aleph Alpha Kolibri 1.1 (model_type "kolibri1_1").

Kolibri 1 (../kolibri1/language.py) with three changes. Ported from
llama.cpp #32891 (src/models/kolibri1.cpp, written at Aleph Alpha) and the
vLLM plugin aleph_alpha_inference 2.0.0.

- Sparse attention in the former full-attention layers. A query scores the
  keys of the first K/V head with the query heads of that group
  (index_num_heads): mean over the heads of ReLU(q.k)/sqrt(head_dim), q and k
  rounded to NVFP4. It attends to its index_topk best keys, the first 8 and
  the last 128. Selectable are the positions [8, p - 128] of a query at p.
  Indexers (index_layers) set the pool the keys are selected from: all of
  them, or, with more than index_pool_size/16 complete blocks of 16 keys, the
  keys of the best blocks, a block scored by its mean key. The other sparse
  layers select from the pool of the indexer before them. K and V are rounded
  to NVFP4, as trained. No RoPE.
- K/V sharing: a sparse layer may read the K/V of an earlier one
  (kv_source_layers); it has no k_proj, v_proj, k_norm.
- mlp_only_layers have a dense SwiGLU MLP instead of the MoE.

Up to index_topk + 136 tokens every key is attended: full attention over
rounded K/V.

Deviation, as in llama.cpp: a block key is NVFP4(mean(NVFP4(keys))); training
averages the unrounded keys. Blocks are scored whether or not their keys are
selectable (sink, recent), also as in llama.cpp; not checked against the vLLM
kernels.

make_cache() returns one cache per layer that owns K/V, in layer order;
layers that read another layer's K/V have none.
"""

import math
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache, RotatingKVCache
from ..mlp import SwiGLUMLP
from ..rope_utils import initialize_rope
from ..switch_layers import SwitchGLU
from .config import ModelConfig

SLIDING = "sliding_attention"
# Transformers says "indexed_attention"; early 1.1 checkpoints "full_attention".
SPARSE = ("indexed_attention", "full_attention")

# Bytes of fp32 routing scores held at once; longer prefills go row by row.
SCORE_BYTES = 1 << 30


def _e4m3_tables():
    """Positive E4M3 values (codes 1..126) and the rounding thresholds between
    neighbours. A tie goes to the even code: `>=` where the upper one is even."""
    grid = [m * 2.0**-9 for m in range(1, 8)]
    grid += [2.0 ** (e - 7) * (1 + m / 8) for e in range(1, 16) for m in range(8)]
    grid = grid[:-1]  # code 127 is NaN, 448 the largest value
    ge, gt = [], []
    for i in range(len(grid) - 1):
        mid = (grid[i] + grid[i + 1]) / 2
        (ge if (i + 2) % 2 == 0 else gt).append(mid)  # upper code is i + 2
    return mx.array(grid), mx.array(ge), mx.array(gt)


_E4M3, _E4M3_GE, _E4M3_GT = _e4m3_tables()


def _round_e2m1(y: mx.array) -> mx.array:
    """Nearest of 0, .5, 1, 1.5, 2, 3, 4, 6 by magnitude, ties to even, saturating."""
    a = mx.abs(y)
    r = mx.where(a <= 5.0, 4.0, 6.0)
    r = mx.where(a < 3.5, 3.0, r)
    r = mx.where(a <= 2.5, 2.0, r)
    r = mx.where(a < 1.75, 1.5, r)
    r = mx.where(a <= 1.25, 1.0, r)
    r = mx.where(a < 0.75, 0.5, r)
    r = mx.where(a <= 0.25, 0.0, r)
    return mx.where(y < 0, -r, r)


def round_nvfp4(x: mx.array) -> mx.array:
    """Round every 16 values along the last axis to NVFP4 and back.

    scale = e4m3(max(amax/6, 2^-9)), value = scale * e2m1(x/scale), both
    nearest even and saturating (ggml_round_nvfp4). The result is exact in
    bf16 and f16.
    """
    shape = x.shape
    g = x.astype(mx.float32).reshape(*shape[:-1], shape[-1] // 16, 16)
    s = mx.maximum(mx.abs(g).max(axis=-1, keepdims=True) / 6.0, 2.0**-9)
    code = (s >= _E4M3_GE).sum(axis=-1) + (s > _E4M3_GT).sum(axis=-1)
    scale = mx.take(_E4M3, code)[..., None]
    return (_round_e2m1(g / scale) * scale).reshape(shape).astype(x.dtype)


def topk_mask(score: mx.array, k: int) -> mx.array:
    """True for the k largest entries above -inf of each row, all if fewer.

    Ties at the boundary go to the lowest index, as in the vLLM kernels.
    """
    live = score > -mx.inf
    if score.shape[-1] <= k:
        return live
    kth = mx.topk(score, k, axis=-1).min(axis=-1, keepdims=True)
    above = score > kth
    tie = (score == kth) & live
    need = k - above.sum(axis=-1, keepdims=True)
    return above | (tie & (mx.cumsum(tie.astype(mx.int32), axis=-1) <= need))


class Attention(nn.Module):
    """Sliding-window attention with RoPE, as in Kolibri 1."""

    def __init__(self, args: ModelConfig):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        dim = args.hidden_size
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            False,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class SparseAttention(nn.Module):
    def __init__(self, args: ModelConfig, own_kv: bool, indexer: bool):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.own_kv = own_kv
        self.indexer = indexer
        self.n_route = args.index_num_heads
        self.topk = args.index_topk
        self.block = args.index_compression
        self.pool_blocks = args.index_pool_size // args.index_compression
        self.n_sink = args.index_num_sink_tokens
        self.n_recent = args.index_num_local_tokens

        dim = args.hidden_size
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        if own_kv:
            self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def _best(self, q, keys, allowed, k):
        """[B, L, M] bool: per query its k best allowed keys.

        q [B, R, L, D] and keys [B, M, D] in fp32, allowed [L, M] or [B, L, M].
        """
        B, R, L, _ = q.shape
        M = keys.shape[1]
        kt = keys.transpose(0, 2, 1)[:, None]
        w = 1.0 / (R * math.sqrt(self.head_dim))
        step = max(1, SCORE_BYTES // (B * R * M * 4))
        out = []
        for s in range(0, L, step):
            score = mx.maximum(q[:, :, s : s + step] @ kt, 0).sum(axis=1) * w
            score = mx.where(allowed[..., s : s + step, :], score, -mx.inf)
            out.append(topk_mask(score, k))
            if step < L:
                mx.eval(out[-1])  # one chunk of scores in memory at a time
        return out[0] if len(out) == 1 else mx.concatenate(out, axis=1)

    def select(self, q, k, pool, offset):
        """Mask [B, 1, L, N] of the keys each query attends to, and the pool.

        q [B, H, L, D] normed queries at positions offset.., k [B, Hkv, N, D]
        rounded keys. pool [B, L, N] bool: the keys of the indexer's best
        blocks, None while it holds every key.
        """
        B, _, L, _ = q.shape
        N = k.shape[2]
        # Nothing to select: fewer selectable keys than index_topk, and no
        # block pool that could exclude some (pool >= topk in the release).
        short = N <= self.topk + self.n_sink + self.n_recent
        if short and N // self.block <= self.pool_blocks:
            return ("causal" if L > 1 else None), None

        pos = mx.arange(offset, N)[:, None]
        n = mx.arange(N)[None]
        cand = (n >= self.n_sink) & (n <= pos - self.n_recent)
        always = (n <= pos) & ~cand

        k_route = k[:, 0].astype(mx.float32)
        q_route = round_nvfp4(q[:, : self.n_route].astype(mx.float32))

        if self.indexer:
            pool = None
            nb = N // self.block
            if nb > self.pool_blocks:
                kb = k_route[:, : nb * self.block].reshape(B, nb, self.block, -1)
                kb = round_nvfp4(kb.mean(axis=2))
                # a block is selectable once it is complete and not ahead of the query
                live = (mx.arange(nb)[None] + 1) * self.block - 1 <= pos
                blocks = self._best(q_route, kb, live, self.pool_blocks)
                pool = mx.repeat(blocks, self.block, axis=-1)
                if nb * self.block < N:
                    tail = mx.zeros((B, L, N - nb * self.block), dtype=mx.bool_)
                    pool = mx.concatenate([pool, tail], axis=-1)

        allowed = cand if pool is None else pool & cand
        mask = self._best(q_route, k_route, allowed, self.topk) | always
        return mask[:, None], pool

    def _gather(self, keys, values, mask):
        """Decode: attend over the selected keys only, not over all under a mask.

        Measured on the 3/6-bit build, M5 Pro: 32 instead of 28 tokens/s at
        96k tokens of context, no difference at 16k. The selected keys come
        first in the partition; with fewer of them the rest stays masked.
        """
        n = self.topk + self.n_sink + self.n_recent
        m = mask[:, 0, 0]
        idx = mx.argpartition(-m.astype(mx.int8), kth=n - 1, axis=-1)[:, :n]
        at = mx.broadcast_to(idx[:, None, :, None], (*keys.shape[:2], n, keys.shape[3]))
        keys = mx.take_along_axis(keys, at, axis=2)
        values = mx.take_along_axis(values, at, axis=2)
        return keys, values, mx.take_along_axis(m, idx, axis=-1)[:, None, None]

    def __call__(self, x, kv=None, pool=None, cache=None):
        """Returns the output, the (K, V) this layer attends over and the pool."""
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)

        if self.own_kv:
            keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
            values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
            keys = round_nvfp4(self.k_norm(keys).transpose(0, 2, 1, 3))
            values = round_nvfp4(values.transpose(0, 2, 1, 3))
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)
            kv = (keys, values)
        keys, values = kv

        mask, pool = self.select(queries, keys, pool, keys.shape[2] - L)
        if L == 1 and isinstance(mask, mx.array):
            keys, values, mask = self._gather(keys, values, mask)
        output = scaled_dot_product_attention(
            queries, keys, values, cache=None, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), kv, pool


class Kolibri1MoE(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob

        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = mx.zeros((args.num_experts,), dtype=mx.float32)
        self.experts = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_experts = SwiGLUMLP(
            args.hidden_size, args.shared_expert_intermediate_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        if isinstance(self.gate, nn.QuantizedLinear):
            logits = self.gate(x).astype(mx.float32)
        else:
            logits = x.astype(mx.float32) @ self.gate.weight.astype(mx.float32).T

        k = self.num_experts_per_tok
        selection = logits + self.expert_bias
        inds = mx.stop_gradient(
            mx.argpartition(-selection, kth=k - 1, axis=-1)[..., :k]
        )
        scores = mx.sigmoid(mx.take_along_axis(logits, inds, axis=-1))
        if self.norm_topk_prob:
            scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)

        y = self.experts(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(x.dtype)
        return y + self.shared_experts(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelConfig, idx: int):
        super().__init__()
        self.use_sliding = args.layer_types[idx] == SLIDING
        if self.use_sliding:
            self.self_attn = Attention(args)
        else:
            source = args.kv_source_layers[idx] if args.kv_source_layers else idx
            self.self_attn = SparseAttention(
                args, own_kv=source == idx, indexer=idx in args.index_layers
            )
        if idx in args.mlp_only_layers:
            self.mlp = SwiGLUMLP(args.hidden_size, args.mlp_only_intermediate_size)
        else:
            self.mlp = Kolibri1MoE(args)

        eps = args.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_attn_norm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_ffn_norm = nn.RMSNorm(args.hidden_size, eps=eps)

    def __call__(self, x, mask=None, cache=None, kv=None, pool=None):
        """Sliding layers return the output; sparse ones (output, (K, V), pool)."""
        a = self.input_layernorm(x)
        if self.use_sliding:
            r = self.self_attn(a, mask, cache)
        else:
            r, kv, pool = self.self_attn(a, kv, pool, cache)
        h = x + self.post_attn_norm(r)
        r = self.mlp(self.post_attention_layernorm(h))
        h = h + self.post_ffn_norm(r)
        return h if self.use_sliding else (h, kv, pool)


class Kolibri1_1Model(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        n = args.num_hidden_layers
        for t in args.layer_types:
            if t != SLIDING and t not in SPARSE:
                raise ValueError(f"kolibri1_1: unknown layer type {t!r}")
        if len(args.layer_types) != n:
            raise ValueError("kolibri1_1: layer_types does not match the layers")

        self.sliding_window = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(n)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.kv_source = list(args.kv_source_layers) or list(range(n))
        # position of each layer's cache in make_cache(), None without own K/V
        self.cache_index, c, indexer = [], 0, False
        for i, layer in enumerate(self.layers):
            own = layer.use_sliding or self.kv_source[i] == i
            self.cache_index.append(c if own else None)
            c += own
            if layer.use_sliding:
                continue
            src = self.kv_source[i]
            if src > i or self.layers[src].use_sliding or self.kv_source[src] != src:
                raise ValueError(f"kolibri1_1: invalid K/V source of layer {i}")
            indexer |= layer.self_attn.indexer
            if not indexer:
                raise ValueError(f"kolibri1_1: sparse layer {i} has no indexer before it")
        self.num_caches = c
        self.swa_cache = next(
            (self.cache_index[i] for i, l in enumerate(self.layers) if l.use_sliding),
            None,
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
    ):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds

        if cache is None:
            cache = [None] * self.num_caches

        swa_mask = None
        if self.swa_cache is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_cache], window_size=self.sliding_window
            )

        kv, pool = {}, None
        for i, layer in enumerate(self.layers):
            c = None if self.cache_index[i] is None else cache[self.cache_index[i]]
            if layer.use_sliding:
                h = layer(h, mask=swa_mask, cache=c)
            else:
                h, kv[i], pool = layer(
                    h, cache=c, kv=kv.get(self.kv_source[i]), pool=pool
                )

        return self.norm(h)


def _dequantize_fp8(weight: mx.array, scale_inv: mx.array, bs: int = 128):
    """128x128 block-FP8 (E4M3 bytes + fp32 inverse scales) to bf16."""
    m, n = weight.shape
    w = mx.from_fp8(weight, dtype=mx.bfloat16)
    pm, pn = (-m) % bs, (-n) % bs
    if pm or pn:
        w = mx.pad(w, ((0, pm), (0, pn)))
    w = w.reshape((m + pm) // bs, bs, (n + pn) // bs, bs)
    w = (w * scale_inv[:, None, :, None]).reshape(m + pm, n + pn)
    return w[:m, :n].astype(mx.bfloat16)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Kolibri1_1Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self, inputs: mx.array, cache=None, inputs_embeds=None, mask=None, **kwargs
    ):
        out = self.model(inputs, cache, inputs_embeds=inputs_embeds)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        weights = {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        # The FP8 release: dequantize to bf16 here. Load it with the
        # checkpoint's quantization_config removed, otherwise the generic
        # loader requantizes it to MXFP8 before this runs.
        for sk in [k for k in weights if k.endswith(".weight_scale_inv")]:
            wk = sk[: -len("_scale_inv")]
            weights[wk] = _dequantize_fp8(weights[wk], weights.pop(sk))

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            bias_key = f"{prefix}.moe.router.expert_bias"
            if bias_key in weights:
                weights[f"{prefix}.mlp.expert_bias"] = weights.pop(bias_key)
            for n in ["gate_proj", "up_proj", "down_proj"]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{n}.{k}" in weights:
                        weights[f"{prefix}.mlp.experts.{n}.{k}"] = mx.stack(
                            [
                                weights.pop(f"{prefix}.mlp.experts.{e}.{n}.{k}")
                                for e in range(self.args.num_experts)
                            ]
                        )
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.model.sliding_window)
                if layer.use_sliding
                else KVCache()
            )
            for i, layer in enumerate(self.layers)
            if self.model.cache_index[i] is not None
        ]

    @property
    def cast_predicate(self):
        def predicate(k):
            return "expert_bias" not in k

        return predicate

    @property
    def quant_predicate(self):
        # The release keeps the router in bf16 (modules_to_not_convert).
        def predicate(path, _):
            return not path.endswith("mlp.gate")

        return predicate
