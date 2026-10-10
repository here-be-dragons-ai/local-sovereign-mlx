#!/usr/bin/env python3
"""Divergence @32: does a quantized build follow the original over several tokens? (sovereign-models#13)

KL divergence and top-1 agreement look one token ahead, always on the
reference's context. Generation builds on its own tokens, so a small early
difference can change everything after it. Here the reference and each build
decode 32 tokens greedily from the same 300 prompts, and we count how often
the build produces exactly the same 32 tokens and where it first leaves the
reference's path.

Unsloth's Divergence-300 prompts are not published; ours follow its
categories, 60 prompts each, from public sources:
  terminal   Terminal-Bench 2.1 task instructions (harborframework/terminal-bench-2.1, Apache-2.0)
  swe        SWE-bench Verified problem statements (princeton-nlp/SWE-bench_Verified)
  math       AIME 2025 and HMMT February 2025 problems (MathArena)
  nonlatin   Belebele passages in Chinese, Arabic, Hindi, Russian and Japanese, to summarise
  longdoc    1,200-word windows (stride 600) of the Wikipedia texts of the quality set, to summarise

    uvx --from pyarrow --with huggingface_hub python measure-divergence.py prepare
    ./measure-divergence.py run    --model <dir> --arm <name> [--loader vlm|lm] [--limit N]
    ./measure-divergence.py report --ref <arm>

Prompts go through each model's chat template with its default thinking
setting (Kolibri: reasoning_effort low; others: the template default). The
model is loaded fully into memory; Kolibri's FP8 reference does not fit and is
not covered yet.
"""

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

DATA = Path("~/src/mlx/divergence").expanduser()
WIKI = Path("~/src/mlx/kolibri-quality/raw").expanduser()
N_PER = 60
STEPS = 32
REVISIONS = {
    "terminal": ("harborframework/terminal-bench-2.1", "main"),
    "swe": ("princeton-nlp/SWE-bench_Verified", "main"),
    "aime": ("MathArena/aime_2025", "main"),
    "hmmt": ("MathArena/hmmt_feb_2025", "main"),
    "belebele": ("facebook/belebele", "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"),
}
NONLATIN = ["zho_Hans", "arb_Arab", "hin_Deva", "rus_Cyrl", "jpn_Jpan"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── prepare ──────────────────────────────────────────────────────────────────


def _parquet(repo, path, rev):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    return pq.read_table(hf_hub_download(repo, path, repo_type="dataset", revision=rev)).to_pylist()


def cmd_prepare(args):
    from huggingface_hub import HfApi, hf_hub_download

    prompts = []
    repo, rev = REVISIONS["terminal"]
    files = HfApi().list_repo_files(repo, repo_type="dataset", revision=rev)
    tasks = sorted(f for f in files if f.endswith("/instruction.md"))[:N_PER]
    for f in tasks:
        text = Path(hf_hub_download(repo, f, repo_type="dataset", revision=rev)).read_text()
        prompts.append(("terminal", f, text.strip()))

    rows = _parquet(*REVISIONS["swe"][:1], "data/test-00000-of-00001.parquet", REVISIONS["swe"][1])
    random.Random(0).shuffle(rows)
    for r in rows[:N_PER]:
        prompts.append(("swe", r["instance_id"],
                        f"Here is an issue from the repository {r['repo']}. Explain the cause and how to fix it.\n\n"
                        + r["problem_statement"].strip()))

    for key in ("aime", "hmmt"):
        repo, rev = REVISIONS[key]
        for r in _parquet(repo, "data/train-00000-of-00001.parquet", rev)[: N_PER // 2]:
            prompts.append(("math", f"{key}-{r.get('problem_idx', len(prompts))}",
                            r["problem"].strip() + "\n\nSolve the problem. Put the final answer in \\boxed{}."))

    repo, rev = REVISIONS["belebele"]
    for lang in NONLATIN:
        seen = []
        for line in open(hf_hub_download(repo, f"data/{lang}.jsonl", repo_type="dataset", revision=rev)):
            p = json.loads(line)["flores_passage"]
            if p not in seen:
                seen.append(p)
            if len(seen) == N_PER // len(NONLATIN):
                break
        for i, p in enumerate(seen):
            prompts.append(("nonlatin", f"{lang}-{i}", "Summarise the following text in two sentences, "
                            "in the language of the text.\n\n" + p))

    words = []
    for f in sorted(WIKI.glob("wiki-*.txt")):
        w = f.read_text().split()
        words += [(f.stem, w[i:i + 1200]) for i in range(0, len(w) - 1200, 600)]
    random.Random(0).shuffle(words)
    for stem, w in words[:N_PER]:
        prompts.append(("longdoc", stem, "Summarise the main points of the following article.\n\n" + " ".join(w)))

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "prompts.jsonl"
    with open(out, "w") as fh:
        for i, (cat, src, text) in enumerate(prompts):
            fh.write(json.dumps({"id": i, "category": cat, "source": src, "prompt": text}, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    (DATA / "prompts.sha256").write_text(sha + "\n")
    counts = defaultdict(int)
    for c, _, _ in prompts:
        counts[c] += 1
    log(f"{len(prompts)} prompts {dict(counts)}, sha256 {sha[:16]}")


# ── run ──────────────────────────────────────────────────────────────────────


def load(path, loader):
    if loader == "lm":
        from mlx_lm import load as lm_load

        model, tok = lm_load(str(path))
        return tok, lambda ids, cache: model(ids, cache=cache), model.make_cache
    from mlx_vlm import load as vlm_load

    model, proc = vlm_load(str(path))
    tok = getattr(proc, "tokenizer", proc)
    lm = model.language_model
    return tok, lambda ids, cache: lm(ids, cache=cache).logits, lm.make_cache


def render(tok, text, model_type):
    kw = {"reasoning_effort": "low"} if model_type == "kolibri1" else {}
    enc = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                  add_generation_prompt=True, **kw)
    return list(enc["input_ids"] if hasattr(enc, "keys") else enc)


def cmd_run(args):
    import mlx.core as mx

    tok, forward, make_cache = load(args.model, args.loader)
    model_type = json.loads((args.model / "config.json").read_text()).get("model_type", "")
    prompts = [json.loads(l) for l in open(DATA / "prompts.jsonl")]
    if args.limit:
        prompts = prompts[:: max(1, len(prompts) // args.limit)]
    (DATA / "runs").mkdir(exist_ok=True)
    out = DATA / "runs" / f"{args.arm}.jsonl"
    t0 = time.time()
    with open(out, "w") as fh:
        for n, p in enumerate(prompts, 1):
            ids = render(tok, p["prompt"], model_type)
            cache = make_cache()
            logits = forward(mx.array([ids]), cache)
            toks = []
            for _ in range(STEPS):
                nxt = int(mx.argmax(logits[0, -1]).item())
                toks.append(nxt)
                logits = forward(mx.array([[nxt]]), cache)
            fh.write(json.dumps({"id": p["id"], "category": p["category"], "prompt_tokens": len(ids),
                                 "tokens": toks}) + "\n")
            if n % 25 == 0:
                log(f"{n}/{len(prompts)} prompts, {time.time() - t0:.0f} s")
    log(f"{args.arm}: {len(prompts)} prompts -> {out}")


# ── report ───────────────────────────────────────────────────────────────────


def cmd_report(args):
    runs = {p.stem: {r["id"]: r for r in map(json.loads, open(p))} for p in sorted((DATA / "runs").glob("*.jsonl"))}
    ref = runs[args.ref]
    result = {"date": time.strftime("%Y-%m-%d"), "prompts_sha256": (DATA / "prompts.sha256").read_text().strip(),
              "reference": args.ref, "steps": STEPS, "arms": {}}
    print(f"| arm | category | prompts | all {STEPS} tokens identical | mean first divergence |")
    print("|---|---|---|---|---|")
    for arm, rows in runs.items():
        if arm == args.ref:
            continue
        cats = defaultdict(list)
        for i, r in rows.items():
            if i not in ref:
                continue
            a, b = ref[i]["tokens"], r["tokens"]
            first = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), STEPS)
            cats[r["category"]].append(first)
            cats["all"].append(first)
        res = {}
        for c, firsts in sorted(cats.items(), key=lambda kv: (kv[0] != "all", kv[0])):
            same = sum(f == STEPS for f in firsts) / len(firsts)
            mean = sum(firsts) / len(firsts)
            res[c] = {"n": len(firsts), "identical": same, "mean_first_divergence": mean}
            print(f"| {arm} | {c} | {len(firsts)} | {same:.1%} | {mean:.1f} |")
        result["arms"][arm] = res
    out = DATA / f"divergence-{result['date']}.json"
    out.write_text(json.dumps(result, indent=1))
    log(f"raw: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    p = sub.add_parser("run")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--loader", choices=["lm", "vlm"], default="vlm")
    p.add_argument("--limit", type=int)
    p = sub.add_parser("report")
    p.add_argument("--ref", required=True)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
