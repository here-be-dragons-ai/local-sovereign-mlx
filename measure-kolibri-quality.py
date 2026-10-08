#!/usr/bin/env python3
"""Measure what the 3-bit quantization costs Kolibri 1, against the FP8 original.

The point of this script is issue #8. The original is block-FP8, 73 GB, and does
not fit on a 48 GB machine. It does not have to: every arm -- the FP8 reference,
the shipped 3/6-bit build and the uniform 3-bit control -- goes through the SAME
layer-streamed forward pass. One decoder layer is read from disk, run over all
sequences of a set, and dropped before the next one is read. The FP8 release
dequantizes lazily in Kolibri1's sanitize, so only the layer being evaluated is
ever materialized (~3 GB in bf16). What is kept is the final normed hidden
state; logits are computed from it at comparison time, with each arm's own LM
head.

    ./measure-kolibri-quality.py prepare                 # texts + benchmarks
    ./measure-kolibri-quality.py check   --ckpt 3bit     # streamed == in-memory?
    ./measure-kolibri-quality.py forward --ckpt fp8      # all sets, resumable
    ./measure-kolibri-quality.py forward --ckpt 3bit
    ./measure-kolibri-quality.py forward --ckpt 3bit-uniform
    ./measure-kolibri-quality.py report                  # KL, PPL, accuracy

--ckpt takes a name from CHECKPOINTS or a path. Data, hidden states and results
live in ~/src/mlx/kolibri-quality/ (--data), not in the repository: the
benchmark data is not ours to redistribute.

STOP THE SERVER FIRST. A streamed layer plus the hidden states of the
benchmark sets need ~10 GB; next to the 33 GB server that does not fit.

THE SETS.
  text         ~60k tokens of Wikipedia prose at pinned revisions, German and
               English, half of it from articles created after Kolibri's
               knowledge cutoff (2026-06-18), cut into 4096-token windows.
               Scored on every position: KL(ref || arm), top-1 agreement,
               perplexity; plus perplexity llama.cpp-style (second half of each
               window only), comparable to `llama-perplexity -c 4096`.
  belebele-*   Belebele deu_Latn / eng_Latn, 900 questions each.
  gmmlu-*      Global-MMLU-Lite de / en, 400 questions each.
               Multiple choice through Kolibri's chat template with
               reasoning_effort=none; the answer is the letter with the highest
               logit among A-D at the first answer position. Accuracy with a
               95% Wilson interval; against the reference also the paired
               agreement and an exact McNemar test.

LOGITS ARE COMPUTED IN FP32 ON THE CPU. On the GPU an fp32 matmul deviates by
~1e-2 (measured during the port), the same order as the KL being measured.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

_HERE = Path(__file__).resolve().parent
MODELS = Path(os.environ.get("MLX_MODELS", "~/src/mlx/models")).expanduser()
CHECKPOINTS = {
    "fp8": MODELS / "Kolibri-1-FP8",
    "3bit": MODELS / "Kolibri-1-MLX-3bit",
    "3bit-uniform": MODELS / "Kolibri-1-MLX-3bit-uniform",
}
REFERENCE = "fp8"
WINDOW = 4096
MIN_WINDOW = 512
LETTERS = "ABCD"
SAVE_EVERY = 10
UA = {"User-Agent": "local-sovereign-mlx/measure-kolibri-quality (issue #8)"}

# Pinned revisions, picked once (2026-10-08): "post" articles were created
# 2026-07-01..09-15, after the knowledge cutoff, longest prose first; "pre"
# articles are the last revision before 2026-01-01. ~60k characters each.
ARTICLES = [
    ("de", "post", 269992864, "Transgeschlechtliche Menschen im Nationalsozialismus"),
    ("de", "post", 269841684, "Kosmographie des anonymen Geographen von Ravenna"),
    ("de", "post", 269565688, "Kapitulation (Roman)"),
    ("de", "post", 270709944, "Francesco Aiello"),
    ("de", "pre", 262053101, "Bodensee"),
    ("en", "post", 1371819861, "Bluefield Blue–Grays"),
    ("en", "post", 1374680605, "Poland's response to the Russian invasion of Ukraine"),
    ("en", "pre", 1328455398, "Photosynthesis"),
    ("en", "pre", 1330398198, "Johann Sebastian Bach"),
]
BELEBELE = ("facebook/belebele", {"de": "data/deu_Latn.jsonl", "en": "data/eng_Latn.jsonl"})
GMMLU = "CohereLabs/Global-MMLU-Lite"

PROMPT = {
    "de": (
        "{context}Frage: {question}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}\n\n"
        "Antworte nur mit dem Buchstaben der richtigen Antwort."
    ),
    "en": (
        "{context}Question: {question}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}\n\n"
        "Answer with the letter of the correct answer only."
    ),
}
PASSAGE = {"de": "Text:\n{p}\n\n", "en": "Passage:\n{p}\n\n"}


def _load_sibling(name, alias):
    spec = importlib.util.spec_from_file_location(alias, _HERE / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ckpt_path(name):
    return CHECKPOINTS.get(name, Path(name).expanduser())


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── prepare ──────────────────────────────────────────────────────────────────


def _get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


class _Prose(HTMLParser):
    """Paragraph text of a rendered Wikipedia page; tables, refs, navboxes dropped."""

    SKIP_TAGS = {"table", "style", "sup", "figure"}
    SKIP_CLASSES = ("mw-editsection", "navbox", "reflist", "references")

    def __init__(self):
        super().__init__()
        self.skip, self.in_p, self.buf, self.out = [], False, [], []

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class") or ""
        if self.skip or tag in self.SKIP_TAGS or any(c in cls for c in self.SKIP_CLASSES):
            self.skip.append(tag)
        elif tag == "p":
            self.in_p, self.buf = True, []

    def handle_endtag(self, tag):
        if self.skip:
            self.skip.pop()
        elif tag == "p" and self.in_p:
            text = " ".join("".join(self.buf).split())
            if len(text) > 80:
                self.out.append(text)
            self.in_p = False

    def handle_data(self, data):
        if self.in_p and not self.skip:
            self.buf.append(data)


def fetch_article(lang, revid):
    q = urllib.parse.urlencode(
        {"action": "parse", "oldid": revid, "prop": "text", "format": "json", "formatversion": 2}
    )
    html = _get_json(f"https://{lang}.wikipedia.org/w/api.php?{q}")["parse"]["text"]
    p = _Prose()
    p.feed(html)
    return "\n\n".join(p.out)


def fetch_gmmlu(lang):
    rows, offset = [], 0
    while True:
        q = urllib.parse.urlencode(
            {"dataset": GMMLU, "config": lang, "split": "test", "offset": offset, "length": 100}
        )
        page = _get_json(f"https://datasets-server.huggingface.co/rows?{q}")
        rows += [r["row"] for r in page["rows"]]
        offset += 100
        if offset >= page["num_rows_total"]:
            return rows


def chat_ids(tok, user):
    enc = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=True,
        add_generation_prompt=True,
        reasoning_effort="none",
    )
    return list(enc["input_ids"] if hasattr(enc, "keys") else enc)


def cmd_prepare(args):
    from transformers import AutoTokenizer

    data = args.data
    (data / "raw").mkdir(parents=True, exist_ok=True)
    (data / "sets").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(ckpt_path("3bit")))
    letter_ids = [tok.encode(c, add_special_tokens=False) for c in LETTERS]
    if any(len(i) != 1 for i in letter_ids) or len({i[0] for i in letter_ids}) != 4:
        sys.exit(f"answer letters are not four distinct single tokens: {letter_ids}")

    # Text set.
    seqs, meta, sources = [], [], []
    for lang, period, revid, title in ARTICLES:
        raw = data / "raw" / f"wiki-{lang}-{revid}.txt"
        if not raw.exists():
            raw.write_text(fetch_article(lang, revid))
        text = raw.read_text()
        sources.append(
            {"lang": lang, "period": period, "revid": revid, "title": title,
             "chars": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest()}
        )
        ids = tok.encode(text, add_special_tokens=False)
        for start in range(0, len(ids), WINDOW):
            window = ids[start : start + WINDOW]
            if len(window) >= MIN_WINDOW:
                seqs.append(window)
                meta.append({"lang": lang, "period": period, "revid": revid})
    (data / "raw" / "llama-perplexity.txt").write_text(
        "\n\n".join((data / "raw" / f"wiki-{l}-{r}.txt").read_text() for l, _, r, _ in ARTICLES)
    )
    _write_set(data, "text", "all", seqs, meta, {"sources": sources})

    # Belebele.
    from huggingface_hub import hf_hub_download

    repo, files = BELEBELE
    for lang, fname in files.items():
        path = hf_hub_download(repo, fname, repo_type="dataset")
        seqs, meta = [], []
        for line in open(path):
            q = json.loads(line)
            user = PROMPT[lang].format(
                context=PASSAGE[lang].format(p=q["flores_passage"]),
                question=q["question"],
                a=q["mc_answer1"], b=q["mc_answer2"], c=q["mc_answer3"], d=q["mc_answer4"],
            )
            seqs.append(chat_ids(tok, user))
            meta.append({"id": f"{q['link']}#{q['question_number']}",
                         "answer": LETTERS[int(q["correct_answer_num"]) - 1]})
        _write_set(data, f"belebele-{lang}", "last", seqs, meta, {"letter_ids": letter_ids})

    # Global-MMLU-Lite.
    for lang in ("de", "en"):
        raw = data / "raw" / f"gmmlu-lite-{lang}.json"
        if not raw.exists():
            raw.write_text(json.dumps(fetch_gmmlu(lang), ensure_ascii=False))
        seqs, meta = [], []
        for q in json.loads(raw.read_text()):
            user = PROMPT[lang].format(
                context="", question=q["question"],
                a=q["option_a"], b=q["option_b"], c=q["option_c"], d=q["option_d"],
            )
            seqs.append(chat_ids(tok, user))
            meta.append({"id": q["sample_id"], "answer": q["answer"].strip(),
                         "cultural": q.get("cultural_sensitivity_label")})
        _write_set(data, f"gmmlu-{lang}", "last", seqs, meta, {"letter_ids": letter_ids})


def _write_set(data, name, score, seqs, meta, extra):
    body = {"name": name, "score": score, "seqs": seqs, "meta": meta, **extra}
    (data / "sets" / f"{name}.json").write_text(json.dumps(body, ensure_ascii=False))
    log(f"set {name}: {len(seqs)} sequences, {sum(map(len, seqs)):,} tokens")


def load_sets(data, names=None):
    out = {}
    for f in sorted((data / "sets").glob("*.json")):
        if names is None or f.stem in names:
            out[f.stem] = json.loads(f.read_text())
    if not out:
        sys.exit(f"no sets in {data / 'sets'}; run `prepare` first")
    return out


# ── forward ──────────────────────────────────────────────────────────────────


def load_lazy(path):
    """The model with lazy weights; the FP8 release goes through the staging copy."""
    from mlx_vlm import load

    config = json.loads((path / "config.json").read_text())
    if "quantization_config" in config:
        conv = _load_sibling("convert-kolibri.py", "_conv")
        staging = conv.make_staging(path)
        try:
            model, _ = load(str(staging), lazy=True)
        finally:
            shutil.rmtree(staging)
    else:
        model, _ = load(str(path), lazy=True)
    return model.language_model


def streamed_forward(lm, seqs, keep_last, partial=None):
    """Final normed hidden states, one decoder layer in memory at a time.

    Returns a list with one array per sequence: [L, H] or, with keep_last, [H].
    `partial` is a path for resume state, written every SAVE_EVERY layers.
    """
    import mlx.core as mx
    from mlx_vlm.models.base import create_attention_mask

    inner = lm.model
    layers = inner.layers
    start = 0
    if partial is not None and partial.exists():
        saved = mx.load(str(partial))
        start = int(saved.pop("_next_layer").item())
        hs = [saved[f"h{i}"] for i in range(len(seqs))]
        log(f"  resuming at layer {start}")
    else:
        hs = []
        for s in seqs:
            h = inner.embed_tokens(mx.array([s]))
            mx.eval(h)
            hs.append(h)
    inner.embed_tokens = None

    masks = {}
    for i in range(start, len(layers)):
        t0 = time.time()
        layer = layers[i]
        for j, h in enumerate(hs):
            L = h.shape[1]
            key = (L, layer.use_sliding)
            if key not in masks:
                masks[key] = create_attention_mask(
                    h, None, window_size=inner.sliding_window if layer.use_sliding else None
                )
            hs[j] = layer(h, masks[key])
            mx.eval(hs[j])
        layers[i] = None
        del layer
        mx.clear_cache()
        log(f"  layer {i:2d}  {time.time() - t0:6.1f} s  "
            f"active {mx.get_active_memory() / 2**30:5.1f} GiB  "
            f"peak {mx.get_peak_memory() / 2**30:5.1f} GiB")
        if partial is not None and (i + 1) % SAVE_EVERY == 0 and i + 1 < len(layers):
            tmp = partial.with_name("tmp-" + partial.name)
            mx.save_safetensors(
                str(tmp), {"_next_layer": mx.array(i + 1), **{f"h{j}": h for j, h in enumerate(hs)}}
            )
            tmp.replace(partial)

    out = []
    for h in hs:
        h = inner.norm(h)[0]
        out.append((h[-1] if keep_last else h).astype(mx.float32))
    mx.eval(out)
    return out


def hidden_path(data, arm, set_name):
    return data / "hidden" / arm / f"{set_name}.safetensors"


def cmd_forward(args):
    import mlx.core as mx

    path = ckpt_path(args.ckpt)
    arm = args.arm or args.ckpt
    sets = load_sets(args.data, args.sets.split(",") if args.sets else None)
    todo = [n for n in sets if not hidden_path(args.data, arm, n).exists()]
    if not todo:
        log(f"{arm}: all sets done")
        return
    for name in todo:
        st = sets[name]
        out = hidden_path(args.data, arm, name)
        out.parent.mkdir(parents=True, exist_ok=True)
        log(f"{arm} / {name}: {len(st['seqs'])} sequences, {sum(map(len, st['seqs'])):,} tokens")
        lm = load_lazy(path)
        hs = streamed_forward(
            lm, st["seqs"], st["score"] == "last", out.with_suffix(".partial.safetensors")
        )
        if st["score"] == "last":
            mx.save_safetensors(str(out), {"h": mx.stack(hs)})
        else:
            mx.save_safetensors(str(out), {"h": mx.concatenate(hs, axis=0)})
        out.with_suffix(".partial.safetensors").unlink(missing_ok=True)
        del lm, hs
        mx.clear_cache()


# ── heads and scoring ────────────────────────────────────────────────────────


def head_weight(path):
    """The arm's LM head as an fp32 matrix [V, H], dequantized if quantized."""
    import mlx.core as mx

    lm = load_lazy(path)
    head = lm.lm_head
    if hasattr(head, "scales"):
        w = mx.dequantize(
            head.weight, head.scales, getattr(head, "biases", None),
            group_size=head.group_size, bits=head.bits, mode=getattr(head, "mode", "affine"),
        )
    else:
        w = head.weight
    w = w.astype(mx.float32)
    mx.eval(w)
    del lm
    return w


def cmd_check(args):
    """Streamed forward against the ordinary in-memory forward on the same checkpoint."""
    import mlx.core as mx
    from mlx_vlm import load

    path = ckpt_path(args.ckpt)
    st = load_sets(args.data, ["text"])["text"]
    seqs = [s[:n] for s, n in zip(st["seqs"][:3], (300, 700, 1500))]

    model, _ = load(str(path))
    ref = []
    for s in seqs:
        h = model.language_model.model(mx.array([s]))[0].astype(mx.float32)
        mx.eval(h)
        ref.append(h)
    del model
    mx.clear_cache()

    streamed = streamed_forward(load_lazy(path), seqs, keep_last=False)
    w = head_weight(path)
    with mx.stream(mx.cpu):
        for a, b in zip(ref, streamed):
            la, lb = a @ w.T, b @ w.T
            pa = mx.softmax(la, axis=-1)
            kl = (pa * (_log_softmax(la) - _log_softmax(lb))).sum(-1)
            agree = (la.argmax(-1) == lb.argmax(-1)).astype(mx.float32).mean()
            mx.eval(kl, agree)
            print(f"  L={a.shape[0]:5d}  max|dh| {mx.abs(a - b).max().item():.3e}  "
                  f"mean KL {kl.mean().item():.3e}  top-1 agree {agree.item():.4f}")


def _log_softmax(x):
    import mlx.core as mx

    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def _wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - r, c + r)


def _mcnemar(b, c):
    """Exact two-sided McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def text_scores(h_ref, h_arm, w_ref, w_arm, seqs, meta, chunk=512):
    """Per scored position: KL(ref||arm), top-1 agreement, NLL of both, group keys."""
    import mlx.core as mx

    rows = []
    off = 0
    with mx.stream(mx.cpu):
        for s, m in zip(seqs, meta):
            L = len(s)
            targets = mx.array(s[1:])
            for a in range(0, L - 1, chunk):
                b = min(a + chunk, L - 1)
                lr = h_ref[off + a : off + b] @ w_ref.T
                la = h_arm[off + a : off + b] @ w_arm.T
                lpr, lpa = _log_softmax(lr), _log_softmax(la)
                kl = (mx.exp(lpr) * (lpr - lpa)).sum(-1)
                t = targets[a:b]
                nll_r = -mx.take_along_axis(lpr, t[:, None], axis=-1)[:, 0]
                nll_a = -mx.take_along_axis(lpa, t[:, None], axis=-1)[:, 0]
                agree = lr.argmax(-1) == la.argmax(-1)
                mx.eval(kl, nll_r, nll_a, agree)
                for i, (k, r, q, g) in enumerate(
                    zip(kl.tolist(), nll_r.tolist(), nll_a.tolist(), agree.tolist())
                ):
                    rows.append((m["lang"], m["period"], a + i >= L // 2, k, r, q, g))
            off += L
    return rows


def _summ(rows):
    import statistics

    kls = sorted(r[3] for r in rows)
    n = len(kls)
    return {
        "n": n,
        "kl_mean": sum(kls) / n,
        "kl_median": statistics.median(kls),
        "kl_p99": kls[min(n - 1, int(0.99 * n))],
        "top1_agree": sum(r[6] for r in rows) / n,
        "ppl_ref": math.exp(sum(r[4] for r in rows) / n),
        "ppl_arm": math.exp(sum(r[5] for r in rows) / n),
    }


def mc_scores(h, w, letter_ids, meta):
    import mlx.core as mx

    ids = mx.array([i[0] for i in letter_ids])
    with mx.stream(mx.cpu):
        logits = h @ w.T
        top = logits.argmax(-1)
        pick = logits[:, ids].argmax(-1)
        mx.eval(top, pick)
    letter_set = set(ids.tolist())
    preds = [LETTERS[p] for p in pick.tolist()]
    return {
        "preds": preds,
        "correct": [p == m["answer"] for p, m in zip(preds, meta)],
        "letter_top1": sum(t in letter_set for t in top.tolist()) / len(preds),
    }


def cmd_report(args):
    import mlx.core as mx

    data = args.data
    arms = [a for a in os.listdir(data / "hidden") if (data / "hidden" / a).is_dir()]
    if REFERENCE not in arms:
        sys.exit("no reference hidden states; run `forward --ckpt fp8` first")
    sets = load_sets(data)
    heads = {a: head_weight(ckpt_path(a)) for a in arms}
    result = {
        "date": time.strftime("%Y-%m-%d"),
        "versions": _versions(),
        "sources": sets["text"].get("sources"),
        "text": {},
        "mc": {},
    }
    per_question = []

    for arm in arms:
        if arm == REFERENCE or not hidden_path(data, arm, "text").exists():
            continue
        h_ref = mx.load(str(hidden_path(data, REFERENCE, "text")))["h"]
        h_arm = mx.load(str(hidden_path(data, arm, "text")))["h"]
        st = sets["text"]
        rows = text_scores(h_ref, h_arm, heads[REFERENCE], heads[arm], st["seqs"], st["meta"])
        groups = {"all": rows}
        for lang in ("de", "en"):
            groups[lang] = [r for r in rows if r[0] == lang]
            for period in ("pre", "post"):
                groups[f"{lang}-{period}"] = [r for r in rows if r[0] == lang and r[1] == period]
        groups["second-half"] = [r for r in rows if r[2]]
        result["text"][arm] = {g: _summ(rs) for g, rs in groups.items() if rs}

    for name, st in sets.items():
        if st["score"] != "last":
            continue
        result["mc"][name] = {}
        per_arm = {}
        for arm in arms:
            p = hidden_path(data, arm, name)
            if p.exists():
                per_arm[arm] = mc_scores(mx.load(str(p))["h"], heads[arm], st["letter_ids"], st["meta"])
        for arm, sc in per_arm.items():
            for m, pred, ok in zip(st["meta"], sc["preds"], sc["correct"]):
                per_question.append({"set": name, "id": m["id"], "arm": arm,
                                     "pred": pred, "answer": m["answer"], "correct": ok})
        ref = per_arm.get(REFERENCE)
        for arm, sc in per_arm.items():
            k, n = sum(sc["correct"]), len(sc["correct"])
            entry = {"n": n, "acc": k / n, "ci95": _wilson(k, n), "letter_top1": sc["letter_top1"]}
            if ref is not None and arm != REFERENCE:
                b = sum(r and not a for r, a in zip(ref["correct"], sc["correct"]))
                c = sum(a and not r for r, a in zip(ref["correct"], sc["correct"]))
                entry.update(
                    same_answer=sum(x == y for x, y in zip(ref["preds"], sc["preds"])) / n,
                    delta_acc=(k - sum(ref["correct"])) / n,
                    mcnemar_p=_mcnemar(b, c),
                    ref_only=b, arm_only=c,
                )
            result["mc"][name][arm] = entry

    out = data / "results" / f"quality-{result['date']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    with open(out.with_name(f"per-question-{result['date']}.jsonl"), "w") as fh:
        for row in per_question:
            fh.write(json.dumps(row) + "\n")
    _print_report(result)
    log(f"raw: {out}")


def _versions():
    import importlib.metadata as md
    import subprocess

    commit = subprocess.run(
        ["git", "-C", str(_HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return {"mlx": md.version("mlx"), "mlx_vlm": md.version("mlx-vlm"), "repo_commit": commit}


def _print_report(r):
    print("\n## KL divergence and perplexity vs FP8 (text set)\n")
    print("| arm | group | tokens | mean KL | median KL | p99 KL | top-1 agree | PPL ref | PPL arm |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm, groups in r["text"].items():
        for g, s in groups.items():
            print(f"| {arm} | {g} | {s['n']:,} | {s['kl_mean']:.4f} | {s['kl_median']:.4f} | "
                  f"{s['kl_p99']:.3f} | {s['top1_agree']:.1%} | {s['ppl_ref']:.3f} | {s['ppl_arm']:.3f} |")
    print("\n## Multiple choice (reasoning_effort=none, letter logits)\n")
    print("| set | arm | n | accuracy | 95% CI | Δ vs FP8 | same answer | McNemar p | letter top-1 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, arms in r["mc"].items():
        for arm, e in arms.items():
            lo, hi = e["ci95"]
            delta = f"{e['delta_acc']:+.1%}" if "delta_acc" in e else ""
            same = f"{e['same_answer']:.1%}" if "same_answer" in e else ""
            p = f"{e['mcnemar_p']:.3f}" if "mcnemar_p" in e else ""
            print(f"| {name} | {arm} | {e['n']} | {e['acc']:.1%} | {lo:.1%}–{hi:.1%} | {delta} | "
                  f"{same} | {p} | {e['letter_top1']:.1%} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=Path("~/src/mlx/kolibri-quality").expanduser())
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    p = sub.add_parser("check")
    p.add_argument("--ckpt", default="3bit")
    p = sub.add_parser("forward")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--arm", help="name of the arm (default: --ckpt)")
    p.add_argument("--sets", help="comma-separated subset of sets")
    sub.add_parser("report")
    args = ap.parse_args()
    args.data = args.data.expanduser()
    {"prepare": cmd_prepare, "check": cmd_check, "forward": cmd_forward, "report": cmd_report}[
        args.cmd
    ](args)


if __name__ == "__main__":
    main()
