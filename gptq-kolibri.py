#!/usr/bin/env python3
"""GPTQ for Kolibri's expert down_proj at 3 bit, layer-streamed from the FP8 release (sovereign-models#28).

The per-layer diagnosis puts most of the 3-bit build's error into the routed
experts' down_proj. GPTQ rounds each weight column with the error of the
columns before it compensated, weighted by the second moment of the layer's
real inputs (H = X^T X per expert), instead of rounding every weight on its
own (RTN). Same bits, same group size 64, same file size.

One streamed FP8 pass: per layer the calibration tokens run through the FP8
layer; for each expert the inputs of its down_proj (SwiGLU of up and gate,
for the tokens routed to it) are accumulated into H; then all 384 experts are
quantized at once. The next layer gets the FP8 layer's output.

Calibration data is disjoint from every measurement set of measure-quality.py:
oasst2 threads not used in the chat set (de, en, fr, es), hermes tool
conversations not used in the tools set, and Europarl in 21 EU languages
(Belebele is avoided: its passages are in the multiple-choice sets).

    uvx --from pyarrow --with huggingface_hub --with fsspec python gptq-kolibri.py calib
    ./gptq-kolibri.py run --out ~/src/mlx/models/Kolibri-1-MLX-3bit-gptq [--layers N]

The output is the shipped build cloned (APFS copy-on-write) plus one shard with
the new down_proj tensors that overrides the cloned ones (as in
kolibri-down4-candidate.py). Per layer it logs the output error on the
calibration inputs, tr(dW H dW^T) relative to tr(W H W^T), for RTN and GPTQ.
"""

import argparse
import importlib.util
import json
import random
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = Path("~/src/mlx/models").expanduser()
DATA = Path("~/src/mlx/kolibri-quality").expanduser()
CALIB = DATA / "gptq" / "calib-texts.json"
SHARD = "model-zz-gptq.safetensors"
GROUP, BITS = 64, 3
EUROPARL = ("Helsinki-NLP/europarl", "ab45e286aef3fb5780067100cb5a1132b52b7949")
EUROPARL_LANGS = ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hu", "it", "lt", "lv",
                  "nl", "pl", "pt", "ro", "sk", "sl", "sv"]
WORDS_PER_LANG = 1100
CHAT_THREADS = {"de": 12, "en": 12, "fr": 6, "es": 6}
TOOL_CONVS = 24
MAX_LEN = 4096


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _mq():
    spec = importlib.util.spec_from_file_location("_mq", HERE / "measure-quality.py")
    mq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mq)
    mq.use_profile("kolibri")
    return mq


# ── calibration texts ────────────────────────────────────────────────────────


def cmd_calib(args):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

    mq = _mq()
    out = {"europarl": {}, "chat": [], "tools": []}
    fs = HfFileSystem()
    pairs = {p.split("/")[-1] for p in fs.ls(f"datasets/{EUROPARL[0]}@{EUROPARL[1]}", detail=False)}
    for lang in EUROPARL_LANGS:
        pair = next((p for p in (f"en-{lang}", f"{lang}-en", f"de-{lang}", f"{lang}-de") if p in pairs), None)
        if lang == "en":
            pair = "de-en"
        files = sorted(f for f in fs.ls(f"datasets/{EUROPARL[0]}@{EUROPARL[1]}/{pair}", detail=False)
                       if f.endswith(".parquet"))
        with fs.open(files[-1]) as fh:  # the last file: far from the start of the corpus
            pf = pq.ParquetFile(fh)
            rows = pf.read_row_group(min(1, pf.num_row_groups - 1)).to_pylist()
        words, text = 0, []
        for r in rows:
            s = r["translation"][lang].strip()
            if len(s.split()) >= 6:  # skip agenda lines such as "see Minutes"
                text.append(s)
                words += len(s.split())
            if words >= WORDS_PER_LANG:
                break
        out["europarl"][lang] = " ".join(text)
        log(f"europarl {lang} ({pair}): {words} words")

    chat_set = json.loads((DATA / "sets" / "chat.json").read_text())
    used = {t for m in chat_set["meta"] for t in m["threads"]}
    repo, rev, fname = mq.OASST2
    path = hf_hub_download(repo, fname, repo_type="dataset", revision=rev)
    for lang, n in CHAT_THREADS.items():
        threads = [msgs for tid, msgs in mq._oasst_threads(path, lang) if tid not in used]
        out["chat"] += [{"lang": lang, "messages": m} for m in threads[-n:]]
    tools_set = json.loads((DATA / "sets" / "tools.json").read_text())
    used = {m["id"] for m in tools_set["meta"]}
    repo, rev, fname = mq.HERMES
    rows = json.load(open(hf_hub_download(repo, fname, repo_type="dataset", revision=rev)))
    random.Random(1).shuffle(rows)
    for r in rows:
        if r["id"] in used:
            continue
        try:
            tools = json.loads(r["tools"])
            msgs = mq._hermes_messages(r["conversations"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        out["tools"].append({"tools": tools, "messages": msgs})
        if len(out["tools"]) == TOOL_CONVS:
            break
    CALIB.parent.mkdir(parents=True, exist_ok=True)
    CALIB.write_text(json.dumps(out, ensure_ascii=False))
    log(f"{CALIB}: {len(out['europarl'])} languages, {len(out['chat'])} chats, {len(out['tools'])} tool conversations")


def calib_seqs(mq, tok):
    c = json.loads(CALIB.read_text())
    seqs = []
    for lang, text in c["europarl"].items():
        seqs.append(tok.encode(text, add_special_tokens=False)[:MAX_LEN])
    for item in c["chat"] + c["tools"]:
        try:
            ids = mq._render(tok, item["messages"], item.get("tools"))
        except Exception:
            continue
        seqs.append(ids[:MAX_LEN])
    return seqs


# ── GPTQ ─────────────────────────────────────────────────────────────────────


def pack(q, bits):
    """Integer codes [..., cols] into mlx's little-endian uint32 bit stream."""
    import mlx.core as mx
    import numpy as np

    q = np.asarray(q, dtype=np.uint64)
    cols = q.shape[-1]
    words = np.zeros(q.shape[:-1] + ((cols * bits + 31) // 32,), dtype=np.uint64)
    for k in range(cols):
        w, o = divmod(k * bits, 32)
        words[..., w] |= (q[..., k] << o) & 0xFFFFFFFF
        if o + bits > 32:
            words[..., w + 1] |= q[..., k] >> (32 - o)
    return mx.array(words.astype(np.uint32))


def gptq(W, H, damp=0.01):
    """W [E, R, C] fp32, H [E, C, C] fp32 -> codes [E, R, C], scales, biases [E, R, C/GROUP]."""
    import mlx.core as mx

    E, R, C = W.shape
    levels = 2 ** BITS - 1
    diag = mx.diagonal(H, axis1=1, axis2=2)
    dead = diag == 0
    H = H + mx.eye(C)[None] * (damp * mx.mean(diag, axis=1, keepdims=True)[..., None] + dead[..., None] * 1.0)
    W = mx.where(dead[:, None, :], 0.0, W)
    with mx.stream(mx.cpu):
        L = mx.linalg.cholesky(H)
        Hinv = mx.linalg.cholesky_inv(L)
        U = mx.linalg.cholesky(Hinv, upper=True)
        mx.eval(U)
    codes, scales, biases = [], [], []
    for g in range(0, C, GROUP):
        Wb = W[:, :, g:g + GROUP]
        lo, hi = mx.min(Wb, axis=-1), mx.max(Wb, axis=-1)
        s = mx.maximum((hi - lo) / levels, 1e-8)
        Ub = U[:, g:g + GROUP, g:g + GROUP]
        Err = []
        Qb = []
        cols = [Wb[:, :, j] for j in range(GROUP)]
        for j in range(GROUP):
            w = cols[j]
            q = mx.clip(mx.round((w - lo) / s), 0, levels)
            dq = q * s + lo
            err = (w - dq) / Ub[:, j, j][:, None]
            for k in range(j + 1, GROUP):
                cols[k] = cols[k] - err * Ub[:, j, k][:, None]
            Qb.append(q)
            Err.append(err)
            if j % 8 == 7:
                mx.eval(cols[j + 1:], Qb, Err)
        Err = mx.stack(Err, axis=-1)
        if g + GROUP < C:
            W = mx.concatenate([W[:, :, :g + GROUP],
                                W[:, :, g + GROUP:] - Err @ U[:, g:g + GROUP, g + GROUP:]], axis=-1)
        codes.append(mx.stack(Qb, axis=-1).astype(mx.uint8))
        scales.append(s)
        biases.append(lo)
        mx.eval(W, codes[-1], s, lo)
    return mx.concatenate(codes, axis=-1), mx.stack(scales, axis=-1), mx.stack(biases, axis=-1)


def output_error(dW, W, H):
    """sum_e tr(dW H dW^T) / sum_e tr(W H W^T)."""
    import mlx.core as mx

    num = mx.sum((dW @ H) * dW)
    den = mx.sum((W @ H) * W)
    return (num / den).item()


def cmd_run(args):
    import mlx.core as mx
    from mlx_vlm.models.base import create_attention_mask
    from transformers import AutoTokenizer

    mq = _mq()
    if args.out.exists() and not args.layers:
        raise SystemExit(f"{args.out} exists")
    tok = AutoTokenizer.from_pretrained(str(mq.ckpt_path(mq.TOKENIZER_ARM)))
    seqs = calib_seqs(mq, tok)
    log(f"calibration: {len(seqs)} sequences, {sum(map(len, seqs)):,} tokens")

    lm = mq.load_lazy(mq.ckpt_path(mq.REFERENCE))
    inner = lm.model
    hs = []
    for s in seqs:
        h = inner.embed_tokens(mx.array([s]))
        mx.eval(h)
        hs.append(h)
    inner.embed_tokens = None
    n_layers = args.layers or len(inner.layers)
    out, report = {}, []
    for i in range(n_layers):
        t0 = time.time()
        layer = inner.layers[i]
        mx.eval(layer.parameters())
        window = inner.sliding_window if layer.use_sliding else None
        moe, ex = layer.mlp, layer.mlp.experts
        E = moe.gate.weight.shape[0]
        k = moe.num_experts_per_tok
        new_hs, xms, all_inds = [], [], []
        for h in hs:
            mask = create_attention_mask(h, None, window_size=window)
            r = layer.self_attn(layer.input_layernorm(h), mask, None)
            hm = h + layer.post_attn_norm(r)
            xm = layer.post_attention_layernorm(hm)[0]
            logits = xm.astype(mx.float32) @ moe.gate.weight.astype(mx.float32).T
            inds = mx.argpartition(-(logits + moe.expert_bias), kth=k - 1, axis=-1)[..., :k]
            o = layer(h, mask)
            mx.eval(xm, inds, o)
            xms.append(xm)
            all_inds.append(inds)
            new_hs.append(o)
        hs = new_hs
        # All calibration tokens of this layer, grouped by expert, once.
        xm = mx.concatenate(xms)
        flat = mx.concatenate(all_inds).flatten()
        order = mx.argsort(flat)
        tok_idx = order // k
        mx.eval(xm, tok_idx)
        se = flat[order].tolist()
        starts = {}
        for pos, e in enumerate(se):
            starts.setdefault(e, [pos, pos])[1] = pos + 1
        Hs, cnt = [None] * E, [0] * E
        for e, (a, b) in starts.items():
            x = xm[tok_idx[a:b]]
            up = x @ ex.up_proj.weight[e].T
            gate = x @ ex.gate_proj.weight[e].T
            z = ex.activation(up, gate).astype(mx.float32)
            Hs[e] = z.T @ z
            cnt[e] = b - a
            if e % 32 == 31:
                mx.eval([h for h in Hs if h is not None])
        C = ex.down_proj.weight.shape[-1]
        H = mx.stack([h if h is not None else mx.zeros((C, C), dtype=mx.float32) for h in Hs])
        counts = mx.array(cnt)
        mx.eval(H)
        del xms, all_inds, xm, Hs
        W = ex.down_proj.weight.astype(mx.float32)
        codes, scales, biases = gptq(W, H)
        dq = (codes.astype(mx.float32) * mx.repeat(scales, GROUP, axis=-1)
              + mx.repeat(biases, GROUP, axis=-1))
        wq, s_rtn, b_rtn = mx.quantize(W, group_size=GROUP, bits=BITS)
        rtn = mx.dequantize(wq, s_rtn, b_rtn, group_size=GROUP, bits=BITS)
        e_rtn, e_gptq = output_error(rtn - W, W, H), output_error(dq - W, W, H)
        prefix = f"model.layers.{i}.mlp.experts.down_proj"
        dtype = mx.bfloat16
        out[f"{prefix}.weight"] = mx.concatenate([pack(codes[a:a + 32], BITS) for a in range(0, E, 32)])
        out[f"{prefix}.scales"] = scales.astype(dtype)
        out[f"{prefix}.biases"] = biases.astype(dtype)
        mx.eval(out[f"{prefix}.scales"], out[f"{prefix}.biases"])
        # The stored bf16 scales and biases must still reproduce the codes' grid.
        check = mx.dequantize(out[f"{prefix}.weight"], out[f"{prefix}.scales"], out[f"{prefix}.biases"],
                              group_size=GROUP, bits=BITS).astype(mx.float32)
        e_stored = output_error(check - W, W, H)
        n_rare = int((counts < 32).sum().item())
        report.append({"layer": i, "rtn": e_rtn, "gptq": e_gptq, "stored": e_stored, "rare_experts": n_rare})
        log(f"layer {i:2d}  output error RTN {e_rtn:.4f}  GPTQ {e_gptq:.4f}  stored {e_stored:.4f}  "
            f"experts with <32 tokens: {n_rare}  {time.time() - t0:.0f} s")
        inner.layers[i] = None
        del layer, W, H, codes, dq, rtn, check
        mx.clear_cache()

    (DATA / "gptq").mkdir(exist_ok=True)
    (DATA / "gptq" / f"report-{time.strftime('%Y-%m-%d')}.json").write_text(json.dumps(report, indent=1))
    if args.layers:
        log("smoke test: no build written")
        return
    subprocess.run(["cp", "-cR", str(MODELS / "Kolibri-1-MLX-3bit"), str(args.out)], check=True)
    mx.save_safetensors(str(args.out / SHARD), out, metadata={"format": "mlx"})
    index_path = args.out / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    for key in out:
        index["weight_map"][key] = SHARD
    tmp = index_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, indent=2))
    tmp.replace(index_path)
    log(f"{args.out}: down_proj GPTQ {BITS} bit in {n_layers} layers, extra shard "
        f"{(args.out / SHARD).stat().st_size / 2**30:.2f} GiB")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calib")
    p = sub.add_parser("run")
    p.add_argument("--out", type=Path, default=MODELS / "Kolibri-1-MLX-3bit-gptq")
    p.add_argument("--layers", type=int, help="only the first N layers, no build (smoke test)")
    args = ap.parse_args()
    {"calib": cmd_calib, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
