#!/usr/bin/env python3
"""MMLU 5-shot the way Unsloth runs it, with a validated harness (sovereign-models#12).

All 14,042 MMLU test questions in the original Hendrycks format, five solved
examples of the same subject in front, no chat template. The prompts come
ready-made from unsloth/studio_mmlu. The answer is the letter whose logit is
highest after "Answer:", taking for each letter the larger of " A" and "A"
(Unsloth: the two tokenizations differ, +0.4 points on Llama).

The 5-shot prefix is the same for every question of a subject, so it runs once
per subject into a KV cache, and each question continues from a copy of that
cache. `check` compares this with running every prompt in full.

    ./measure-mmlu.py prepare
    ./measure-mmlu.py check --model <dir> [--loader lm|vlm]     # shared == full?
    ./measure-mmlu.py run   --model <dir> --arm <name> [--loader lm|vlm] [--limit N]
    ./measure-mmlu.py report --ref <arm> [--size-gb arm=GB ...]

Harness validation (Unsloth's first step): run Llama-3.1-8B-Instruct in bf16
with --loader lm; its 5-shot MMLU should come out near 68.2% (Unsloth's
figure for a correct implementation; naive ones got 35%).

The model is loaded completely into memory, so this needs a build that fits
(Kolibri's FP8 release does not; its 3/6-bit build does). Run with the venv
that knows the model (Apertus: ~/src/mlx/.venv-apertus). No server may run.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

DATA = Path("~/src/mlx/mmlu5").expanduser()
DATASET = "unsloth/studio_mmlu"
LETTERS = "ABCD"
_HERE = Path(__file__).resolve().parent


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── prepare ──────────────────────────────────────────────────────────────────


def cmd_prepare(args):
    """Download the parquet file of unsloth/studio_mmlu (needs pyarrow:
    `uvx --from pyarrow python measure-mmlu.py prepare`)."""
    import pyarrow.parquet as pq

    DATA.mkdir(parents=True, exist_ok=True)
    raw = DATA / "studio_mmlu.parquet"
    if not raw.exists():
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/data/train-00000-of-00001.parquet"
        with urllib.request.urlopen(url, timeout=300) as r:
            raw.write_bytes(r.read())
    rows = pq.read_table(raw).to_pylist()
    out = DATA / "questions.jsonl"
    with open(out, "w") as fh:
        for i, r in enumerate(rows):
            fh.write(json.dumps({"id": i, "subject": r["Subject"], "category": r["Category"],
                                 "prefix": r["5_shot"], "question": r["Q"], "answer": r["A"].strip()}) + "\n")
    sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    (DATA / "questions.sha256").write_text(sha + "\n")
    log(f"{len(rows)} questions, {len({r['Subject'] for r in rows})} subjects, parquet sha256 {sha[:16]}")


def questions(limit=None):
    qs = [json.loads(l) for l in open(DATA / "questions.jsonl")]
    if limit:
        # Spread a limited run over all subjects.
        by = defaultdict(list)
        for q in qs:
            by[q["subject"]].append(q)
        per = max(1, limit // len(by))
        qs = [q for s in by.values() for q in s[:per]]
    return qs


# ── model ────────────────────────────────────────────────────────────────────


def load(path, loader):
    if loader == "lm":
        from mlx_lm import load as lm_load

        model, tok = lm_load(str(path))
        return model, tok, lambda ids, cache: model(ids, cache=cache), model.make_cache
    from mlx_vlm import load as vlm_load

    model, proc = vlm_load(str(path))
    tok = getattr(proc, "tokenizer", proc)
    lm = model.language_model
    return model, tok, lambda ids, cache: lm(ids, cache=cache).logits, lm.make_cache


def letter_ids(tok):
    """Token ids per letter, for " A" and "A" where each is a single token."""
    out = []
    for c in LETTERS:
        ids = {t[0] for t in (tok.encode(" " + c, add_special_tokens=False),
                              tok.encode(c, add_special_tokens=False)) if len(t) == 1}
        if not ids:
            raise SystemExit(f"letter {c} is not a single token")
        out.append(sorted(ids))
    return out


def letter_scores(logits, letters):
    import mlx.core as mx

    last = logits[0, -1].astype(mx.float32)
    return [max(last[i].item() for i in ids) for ids in letters]


def encode_split(tok, q):
    """Prefix and suffix ids such that prefix + suffix == the full prompt's ids."""
    full = tok.encode(q["prefix"] + q["question"])
    pre = tok.encode(q["prefix"])
    if full[: len(pre)] == pre:
        return pre, full[len(pre):]
    return None, full


def run(model_path, loader, qs, shared=True):
    import mlx.core as mx

    _, tok, forward, make_cache = load(model_path, loader)
    letters = letter_ids(tok)
    by = defaultdict(list)
    for q in qs:
        by[q["subject"]].append(q)
    out, fallback, t0 = [], 0, time.time()
    for n, (subject, group) in enumerate(by.items(), 1):
        base, base_ids = None, None
        for q in group:
            pre, suf = encode_split(tok, q)
            if shared and pre is not None:
                if base_ids != pre:
                    base, base_ids = make_cache(), pre
                    mx.eval(forward(mx.array([pre]), base))
                cache = copy.deepcopy(base)
                logits = forward(mx.array([suf]), cache)
            else:
                fallback += pre is None
                ids = (pre + suf) if pre is not None else suf
                logits = forward(mx.array([ids]), make_cache())
            scores = letter_scores(logits, letters)
            pred = LETTERS[max(range(4), key=lambda i: scores[i])]
            out.append({"id": q["id"], "subject": subject, "category": q["category"], "pred": pred,
                        "answer": q["answer"], "correct": pred == q["answer"], "scores": scores})
        if n % 10 == 0 or n == len(by):
            acc = sum(r["correct"] for r in out) / len(out)
            log(f"{n}/{len(by)} subjects, {len(out)} questions, acc {acc:.1%}, {time.time() - t0:.0f} s")
    return out, fallback


def cmd_check(args):
    qs = questions(limit=args.limit or 57)
    a, _ = run(args.model, args.loader, qs, shared=True)
    b, fb = run(args.model, args.loader, qs, shared=False)
    same = sum(x["pred"] == y["pred"] for x, y in zip(a, b))
    dmax = max(abs(s - t) for x, y in zip(a, b) for s, t in zip(x["scores"], y["scores"]))
    log(f"shared vs full: same answer {same}/{len(a)}, max |Δ letter logit| {dmax:.4f}, "
        f"prompts that did not split cleanly: {fb}")


def cmd_run(args):
    qs = questions(args.limit)
    res, fb = run(args.model, args.loader, qs)
    (DATA / "runs").mkdir(exist_ok=True)
    out = DATA / "runs" / f"{args.arm}.jsonl"
    with open(out, "w") as fh:
        for r in res:
            fh.write(json.dumps(r) + "\n")
    acc = sum(r["correct"] for r in res) / len(res)
    log(f"{args.arm}: {acc:.2%} on {len(res)} questions ({fb} without a clean prefix split) -> {out}")


# ── report ───────────────────────────────────────────────────────────────────


def cmd_report(args):
    spec = importlib.util.spec_from_file_location("_mq", _HERE / "measure-quality.py")
    mq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mq)
    runs = {p.stem: {r["id"]: r for r in map(json.loads, open(p))} for p in sorted((DATA / "runs").glob("*.jsonl"))}
    sizes = dict(s.split("=") for s in args.size_gb or [])
    ref = runs.get(args.ref)
    result = {"date": time.strftime("%Y-%m-%d"), "questions_sha256": (DATA / "questions.sha256").read_text().strip(),
              "arms": {}}
    print("| arm | n | accuracy | 95% CI | Δ vs ref (95% CI) | equivalent ±1 pp | flips −/+ | efficiency |")
    print("|---|---|---|---|---|---|---|---|")
    for arm, rows in runs.items():
        ids = sorted(rows if ref is None else set(rows) & set(ref))
        k, n = sum(rows[i]["correct"] for i in ids), len(ids)
        e = {"n": n, "acc": k / n, "ci95": mq._wilson(k, n)}
        cats = defaultdict(list)
        for i in ids:
            cats[rows[i]["category"]].append(rows[i]["correct"])
        e["by_category"] = {c: sum(v) / len(v) for c, v in sorted(cats.items())}
        if arm in sizes:
            e["efficiency"] = (100 * e["acc"] - 25) / float(sizes[arm])
        if ref is not None and arm != args.ref:
            e.update(mq._paired([(ref[i]["correct"], ref[i]["pred"]) for i in ids],
                                [(rows[i]["correct"], rows[i]["pred"]) for i in ids]))
        result["arms"][arm] = e
        lo, hi = e["ci95"]
        d = (f"{e['delta_acc'] * 100:+.1f} ({e['delta_ci95'][0] * 100:+.1f} to {e['delta_ci95'][1] * 100:+.1f})"
             if "delta_acc" in e else "")
        eq = ("yes" if e["equivalent"] else "no") if "equivalent" in e else ""
        fl = f"{e['ref_only']}/{e['arm_only']}" if "flips" in e else ""
        ef = f"{e['efficiency']:.2f}/GB" if "efficiency" in e else ""
        print(f"| {arm} | {n} | {e['acc']:.1%} | {lo:.1%}–{hi:.1%} | {d} | {eq} | {fl} | {ef} |")
    out = DATA / f"mmlu5-{result['date']}.json"
    out.write_text(json.dumps(result, indent=1))
    log(f"raw: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    for name in ("check", "run"):
        p = sub.add_parser(name)
        p.add_argument("--model", type=Path, required=True)
        p.add_argument("--loader", choices=["lm", "vlm"], default="vlm")
        p.add_argument("--limit", type=int)
        if name == "run":
            p.add_argument("--arm", required=True)
    p = sub.add_parser("report")
    p.add_argument("--ref", required=True)
    p.add_argument("--size-gb", nargs="*")
    args = ap.parse_args()
    {"prepare": cmd_prepare, "check": cmd_check, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
