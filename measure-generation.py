#!/usr/bin/env python3
"""Generation benchmarks: does a quantized build still reason, follow instructions and find things in long context? (sovereign-models#17)

Multiple choice and next-token distributions are measured on the reference's
context. Quantization damage tends to show first when the model generates
on its own: in reasoning chains, in long context, in following instructions.
Three tasks, all scored by rules, greedy decoding, the same prompts for the
reference and every build:

  mgsm     MGSM (juletxara/mgsm, CC BY-SA 4.0): 250 grade-school maths
           problems per language, de en fr es, solved step by step; scored on
           the final number.
  ifeval   Verifiable instructions in the style of IFEval, German and English,
           generated here from a fixed seed: word limits, bullet counts, JSON,
           no commas, lowercase, keywords, closing phrase, title, paragraphs,
           postscript, placeholders. Scored per prompt (all instructions
           followed) and per instruction.
  ruler    Long context in the style of RULER, on the Wikipedia texts of the
           quality set (German and English haystacks): one needle, one of four
           keys, all three values of one key, and variable tracking over four
           hops; 8k and 32k tokens by default (--lengths for 64k).

A judge comparison (reference vs build, judged by a third model) is not in
here; it needs a judge model chosen by hand.

    uvx --from pyarrow --with huggingface_hub python measure-generation.py prepare
    ./measure-generation.py run    --model <dir> --arm <name> [--loader vlm|lm] [--tasks mgsm,ifeval,ruler] [--limit N]
    ./measure-generation.py report --ref <arm>

Prompts go through each model's chat template with thinking off (Kolibri:
reasoning_effort none; Apertus: the template default). Generation runs in
mlx-vlm's (or mlx-lm's) batch generator, greedy; outputs are stored raw and
scored only in `report`, so a scoring fix needs no new run. The model is loaded
fully into memory; Kolibri's FP8 reference does not fit, so Kolibri builds are
compared with each other and with published scores. No server may run.
"""

import argparse
import hashlib
import importlib.util
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

DATA = Path("~/src/mlx/generation").expanduser()
WIKI = Path("~/src/mlx/kolibri-quality/raw").expanduser()
MGSM = ("juletxara/mgsm", "b2f13d426afe3be8d69a7e739b36724db8b66bbc")
MGSM_LANGS = ("de", "en", "fr", "es")
IFEVAL_PER_LANG = 120
RULER_PER_TASK = 5  # per task, language and length
MAX_TOKENS = {"mgsm": 768, "ifeval": 1024, "ruler": 96}
_HERE = Path(__file__).resolve().parent


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── MGSM ─────────────────────────────────────────────────────────────────────

MGSM_PROMPT = {
    "de": "Löse diese Aufgabe Schritt für Schritt. Beende deine Antwort mit einer Zeile "
          "„Antwort: <Zahl>“.\n\n{q}",
    "en": "Solve this problem step by step. End your response with a line \"Answer: <number>\".\n\n{q}",
    "fr": "Résous ce problème étape par étape. Termine ta réponse par une ligne "
          "« Réponse : <nombre> ».\n\n{q}",
    "es": "Resuelve este problema paso a paso. Termina tu respuesta con una línea "
          "«Respuesta: <número>».\n\n{q}",
}
MGSM_KEY = {"de": "Antwort", "en": "Answer", "fr": "Réponse", "es": "Respuesta"}


def prepare_mgsm():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    out = []
    for lang in MGSM_LANGS:
        path = hf_hub_download(MGSM[0], f"{lang}/test-00000-of-00001.parquet", repo_type="dataset",
                               revision=MGSM[1])
        for i, r in enumerate(pq.read_table(path).to_pylist()):
            out.append({"task": "mgsm", "lang": lang, "id": f"mgsm-{lang}-{i}",
                        "prompt": MGSM_PROMPT[lang].format(q=r["question"].strip()),
                        "answer": r["answer_number"]})
    return out


_NUM = re.compile(r"-?\d[\d.,   ]*")


def _number(s):
    """'1.234' / '1,234' / '1 234' as thousands, '2,5' / '2.5' as decimals."""
    s = s.strip().rstrip(".,").replace(" ", " ").replace(" ", " ")
    if re.fullmatch(r"-?\d{1,3}([., ]\d{3})+", s):
        s = re.sub(r"[., ]", "", s)
    s = s.replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def score_mgsm(item, text):
    m = list(re.finditer(MGSM_KEY[item["lang"]] + r"\s*:\s*\**\s*\$?(" + _NUM.pattern + ")", text))
    nums = [m[-1].group(1)] if m else _NUM.findall(text)[-1:]
    pred = _number(nums[0]) if nums else None
    return pred is not None and abs(pred - float(item["answer"])) < 1e-6, "" if pred is None else f"{pred:g}"


# ── IFEval-style instructions ────────────────────────────────────────────────

TOPICS = {
    "en": ["the history of the bicycle", "why bees matter for agriculture", "how a heat pump works",
           "the benefits of public libraries", "a weekend trip to the mountains", "learning a second language",
           "the invention of the printing press", "how to plan a small vegetable garden",
           "the role of rivers in early cities", "what makes a good team meeting", "the night sky in winter",
           "recycling glass", "the life of a lighthouse keeper", "how bread is made",
           "choosing a first programming language", "the water cycle", "running a village festival",
           "the advantages of trains over planes", "how vaccines are tested", "the craft of watchmaking"],
    "de": ["die Geschichte des Fahrrads", "warum Bienen für die Landwirtschaft wichtig sind",
           "wie eine Wärmepumpe funktioniert", "den Nutzen öffentlicher Bibliotheken",
           "einen Wochenendausflug in die Berge", "das Lernen einer zweiten Sprache",
           "die Erfindung des Buchdrucks", "die Planung eines kleinen Gemüsegartens",
           "die Rolle von Flüssen in frühen Städten", "was eine gute Teambesprechung ausmacht",
           "den Sternenhimmel im Winter", "das Recycling von Glas", "das Leben eines Leuchtturmwärters",
           "wie Brot gebacken wird", "die Wahl einer ersten Programmiersprache", "den Wasserkreislauf",
           "die Organisation eines Dorffests", "die Vorteile von Zügen gegenüber Flugzeugen",
           "wie Impfstoffe getestet werden", "das Handwerk des Uhrmachers"],
}
TASK_TEXT = {"en": "Write a short text about {t}.", "de": "Schreibe einen kurzen Text über {t}."}
KEYWORDS = {"en": ["important", "simple", "future", "people", "water"],
            "de": ["wichtig", "einfach", "Zukunft", "Menschen", "Wasser"]}
PHRASES = {"en": ["Is there anything else I can help with?", "That is all for today."],
           "de": ["Gibt es noch etwas, wobei ich helfen kann?", "Das ist alles für heute."]}

# kind -> (English, German) instruction text; parameters filled in by _instruction.
INSTR = {
    "max_words": ("Answer in at most {n} words.", "Antworte in höchstens {n} Wörtern."),
    "min_words": ("Answer in at least {n} words.", "Antworte in mindestens {n} Wörtern."),
    "bullets": ("Your answer must contain exactly {n} bullet points in Markdown, each starting with \"* \".",
                "Deine Antwort muss genau {n} Aufzählungspunkte in Markdown enthalten, jeder beginnt mit \"* \"."),
    "no_commas": ("Do not use any commas in your entire response.",
                  "Verwende in deiner gesamten Antwort keine Kommas."),
    "lowercase": ("Your entire response must be in lowercase letters; no capital letters are allowed.",
                  "Deine gesamte Antwort muss in Kleinbuchstaben geschrieben sein; Großbuchstaben sind nicht erlaubt."),
    "keyword": ("Use the word \"{kw}\" at least {n} times.", "Verwende das Wort „{kw}“ mindestens {n} Mal."),
    "end_phrase": ("Finish your response with this exact phrase: \"{p}\". No other words may follow it.",
                   "Beende deine Antwort mit genau diesem Satz: „{p}“. Danach darf nichts mehr folgen."),
    "title": ("Give your answer a title wrapped in double angle brackets, such as <<my title>>.",
              "Gib deiner Antwort einen Titel in doppelten spitzen Klammern, zum Beispiel <<mein Titel>>."),
    "json": ("Wrap your entire output in JSON format. You may use Markdown code fences.",
             "Gib deine gesamte Ausgabe im JSON-Format aus. Du darfst Markdown-Codeblöcke verwenden."),
    "paragraphs": ("Write exactly {n} paragraphs, separated from each other by the Markdown divider ***.",
                   "Schreibe genau {n} Absätze, getrennt durch den Markdown-Trenner ***."),
    "postscript": ("At the end of your response, add a postscript starting with P.S.",
                   "Füge am Ende deiner Antwort ein Postskriptum hinzu, das mit P.S. beginnt."),
    "placeholders": ("Include at least {n} placeholders in square brackets, such as [address].",
                     "Füge mindestens {n} Platzhalter in eckigen Klammern ein, zum Beispiel [Adresse]."),
}
# Pairs that cannot both be followed, or that make each other's check meaningless.
CONFLICTS = {frozenset(p) for p in [
    ("json", "bullets"), ("json", "paragraphs"), ("json", "end_phrase"), ("json", "title"),
    ("json", "postscript"), ("json", "lowercase"), ("json", "no_commas"), ("json", "placeholders"),
    ("lowercase", "end_phrase"), ("lowercase", "postscript"), ("lowercase", "keyword"),
    ("max_words", "min_words"), ("bullets", "paragraphs"), ("end_phrase", "postscript"),
    ("max_words", "paragraphs"), ("max_words", "bullets"),
]}


def _instruction(kind, lang, rnd):
    args = {"n": {"max_words": rnd.choice([50, 80, 120]), "min_words": rnd.choice([250, 350]),
                  "bullets": rnd.choice([3, 4, 5]), "keyword": rnd.choice([2, 3]),
                  "paragraphs": rnd.choice([2, 3, 4]), "placeholders": rnd.choice([2, 3])}.get(kind)}
    if kind == "keyword":
        args["kw"] = rnd.choice(KEYWORDS[lang])
    if kind == "end_phrase":
        args["p"] = rnd.choice(PHRASES[lang])
    text = INSTR[kind][lang == "de"].format(**{k: v for k, v in args.items() if v is not None})
    return text, {"kind": kind, **{k: v for k, v in args.items() if v is not None}}


def prepare_ifeval():
    rnd = random.Random(17)
    out = []
    kinds = sorted(INSTR)
    for lang in ("en", "de"):
        for i in range(IFEVAL_PER_LANG):
            topic = TOPICS[lang][i % len(TOPICS[lang])]
            chosen = [rnd.choice(kinds)]
            if rnd.random() < 0.6:
                rest = [k for k in kinds if k not in chosen
                        and all(frozenset((k, c)) not in CONFLICTS for c in chosen)]
                chosen.append(rnd.choice(rest))
            texts, checks = zip(*(_instruction(k, lang, rnd) for k in chosen))
            out.append({"task": "ifeval", "lang": lang, "id": f"ifeval-{lang}-{i}",
                        "prompt": TASK_TEXT[lang].format(t=topic) + " " + " ".join(texts),
                        "checks": list(checks)})
    return out


def _strip_fences(text):
    t = text.strip()
    m = re.fullmatch(r"```[a-zA-Z]*\s*\n?(.*?)\n?```", t, re.S)
    return m.group(1) if m else t


def _check(c, text):
    k, words = c["kind"], text.split()
    if k == "max_words":
        return len(words) <= c["n"]
    if k == "min_words":
        return len(words) >= c["n"]
    if k == "bullets":
        return len(re.findall(r"^\s*[*-] ", text, re.M)) == c["n"]
    if k == "no_commas":
        return "," not in text
    if k == "lowercase":
        return text == text.lower()
    if k == "keyword":
        return len(re.findall(r"\b" + re.escape(c["kw"]) + r"\b", text, re.I)) >= c["n"]
    if k == "end_phrase":
        return text.strip().endswith(c["p"])
    if k == "title":
        return re.search(r"<<[^\n<>]+>>", text) is not None
    if k == "json":
        try:
            json.loads(_strip_fences(text))
            return True
        except ValueError:
            return False
    if k == "paragraphs":
        parts = [p for p in re.split(r"\s*\*\*\*\s*", text.strip()) if p.strip()]
        return len(parts) == c["n"] and text.count("***") == c["n"] - 1
    if k == "postscript":
        return re.search(r"^\s*P\.S\.", text, re.M) is not None
    if k == "placeholders":
        return len(re.findall(r"\[[^\[\]\n]+\]", text)) >= c["n"]
    raise ValueError(k)


def score_ifeval(item, text):
    res = [_check(c, text) for c in item["checks"]]
    return all(res), "".join("1" if r else "0" for r in res)


# ── RULER-style long context ─────────────────────────────────────────────────

RULER_TEXT = {
    "en": {"needle": "The special magic number for {k} is: {v}.",
           "q_single": "What is the special magic number for {k} mentioned in the text above? Answer with the number only.",
           "q_multi": "What are all the special magic numbers for {k} mentioned in the text above? List them all.",
           "vt_start": "VAR {a} = {v}.", "vt_hop": "VAR {b} = VAR {a}.",
           "q_vt": "Find all variables that are assigned the value {v} in the text above, directly or through "
                   "other variables. List their names only.",
           "intro": "Below is a long text. Somewhere in it are a few short statements; read carefully.\n\n"},
    "de": {"needle": "Die besondere magische Zahl für {k} lautet: {v}.",
           "q_single": "Wie lautet die besondere magische Zahl für {k}, die im Text oben genannt wird? Antworte nur mit der Zahl.",
           "q_multi": "Welche besonderen magischen Zahlen für {k} werden im Text oben genannt? Nenne alle.",
           "vt_start": "VAR {a} = {v}.", "vt_hop": "VAR {b} = VAR {a}.",
           "q_vt": "Finde alle Variablen, denen im Text oben der Wert {v} zugewiesen wird, direkt oder über "
                   "andere Variablen. Nenne nur ihre Namen.",
           "intro": "Unten steht ein langer Text. Irgendwo darin stehen einige kurze Aussagen; lies genau.\n\n"},
}
KEYS = ["amber-falcon", "quiet-river", "copper-lantern", "silver-meadow", "northern-orchard",
        "hidden-harbor", "velvet-canyon", "frozen-garden", "crimson-bridge", "golden-thistle"]
# Words per token, roughly, for sizing the haystack; the actual token count is
# recorded per model at run time.
WORDS_PER_TOKEN = {"en": 0.70, "de": 0.54}


def _haystack(lang):
    words = []
    for f in sorted(WIKI.glob("wiki-*.txt")):
        if f"-{lang}-" in f.name or f.name.startswith(f"wiki-{lang}"):
            words += f.read_text().split()
    if not words:
        raise SystemExit(f"no {lang} Wikipedia texts in {WIKI} (measure-quality.py prepare)")
    return words


def _insert(words, sentences, rnd):
    """Insert sentences at random depths (10-90%), at sentence boundaries, in order."""
    ends = [i + 1 for i, w in enumerate(words) if w.endswith(".")]
    lo, hi = int(0.1 * len(words)), int(0.9 * len(words))
    ends = [e for e in ends if lo <= e <= hi] or [len(words) // 2]
    spots = sorted(rnd.sample(ends, min(len(sentences), len(ends))))
    out, last = [], 0
    for spot, s in zip(spots, sentences):
        out += words[last:spot] + s.split()
        last = spot
    return " ".join(out + words[last:])


def prepare_ruler(lengths):
    rnd = random.Random(23)
    out = []
    for lang in ("en", "de"):
        hay = _haystack(lang)
        T = RULER_TEXT[lang]
        for length in lengths:
            n_words = int(length * WORDS_PER_TOKEN[lang])
            for task in ("single", "multikey", "multivalue", "vt"):
                for i in range(RULER_PER_TASK):
                    start = rnd.randrange(0, max(1, len(hay) - n_words))
                    words = (hay * (1 + n_words // len(hay)))[start:start + n_words]
                    keys = rnd.sample(KEYS, 4)
                    vals = [str(rnd.randrange(1_000_000, 9_999_999)) for _ in range(4)]
                    if task == "single":
                        sents, q, ans = [T["needle"].format(k=keys[0], v=vals[0])], \
                            T["q_single"].format(k=keys[0]), [vals[0]]
                    elif task == "multikey":
                        sents = [T["needle"].format(k=k, v=v) for k, v in zip(keys, vals)]
                        rnd.shuffle(sents)
                        q, ans = T["q_single"].format(k=keys[2]), [vals[2]]
                    elif task == "multivalue":
                        sents = [T["needle"].format(k=keys[0], v=v) for v in vals[:3]]
                        q, ans = T["q_multi"].format(k=keys[0]), vals[:3]
                    else:
                        names = ["".join(rnd.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(5))
                                 for _ in range(10)]
                        chain, other = names[:5], names[5:]
                        sents = [T["vt_start"].format(a=chain[0], v=vals[0])]
                        sents += [T["vt_hop"].format(a=a, b=b) for a, b in zip(chain, chain[1:])]
                        sents += [T["vt_start"].format(a=other[0], v=vals[1])]
                        sents += [T["vt_hop"].format(a=a, b=b) for a, b in zip(other, other[1:3])]
                        # Chain order matters, so distractors are interleaved but each chain stays ordered.
                        merged = []
                        a, b = list(sents[:5]), list(sents[5:])
                        while a or b:
                            src = a if (a and (not b or rnd.random() < 0.6)) else b
                            merged.append(src.pop(0))
                        sents, q, ans = merged, T["q_vt"].format(v=vals[0]), chain
                    text = T["intro"] + _insert(words, sents, rnd) + "\n\n" + q
                    out.append({"task": "ruler", "lang": lang, "id": f"ruler-{lang}-{length}-{task}-{i}",
                                "length": length, "kind": task, "prompt": text, "answer": ans})
    return out


def score_ruler(item, text):
    hit = [a for a in item["answer"] if re.search(r"(?<![\w])" + re.escape(a) + r"(?![\w])", text)]
    return len(hit) == len(item["answer"]), str(len(hit))


SCORE = {"mgsm": score_mgsm, "ifeval": score_ifeval, "ruler": score_ruler}


def cmd_prepare(args):
    items = prepare_mgsm() + prepare_ifeval() + prepare_ruler([int(x) for x in args.lengths.split(",")])
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "prompts.jsonl"
    with open(out, "w") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    (DATA / "prompts.sha256").write_text(sha + "\n")
    counts = defaultdict(int)
    for it in items:
        counts[it["task"]] += 1
    log(f"{len(items)} prompts {dict(counts)}, sha256 {sha[:16]}")


# ── run ──────────────────────────────────────────────────────────────────────


def load(path, loader):
    config = json.loads((path / "config.json").read_text())
    eos = config.get("eos_token_id") or config.get("text_config", {}).get("eos_token_id")
    stops = set(eos if isinstance(eos, list) else [eos])
    if loader == "lm":
        from mlx_lm import load as lm_load
        from mlx_lm.generate import BatchGenerator

        model, tok = lm_load(str(path))
        stops |= set(getattr(tok, "eos_token_ids", []) or [])

        def make(max_tokens, batch):
            gen = BatchGenerator(model, max_tokens=max_tokens, stop_tokens=[[s] for s in stops],
                                 completion_batch_size=batch, prefill_batch_size=min(batch, 8))
            return gen, lambda ids, n: gen.insert([ids], [n])[0]
        return tok, config.get("model_type", ""), make
    from mlx_vlm import load as vlm_load
    from mlx_vlm.generate import BatchGenerator

    model, proc = vlm_load(str(path))
    tok = getattr(proc, "tokenizer", proc)

    def make(max_tokens, batch):
        import mlx.core as mx

        gen = BatchGenerator(model, proc, max_tokens=max_tokens, stop_tokens=stops, greedy_sampling=True,
                             completion_batch_size=batch, prefill_batch_size=min(batch, 8))

        def insert(ids, n):
            # The batch generator prefills from embeddings, as the server does.
            emb = model.get_input_embeddings(mx.array([ids]), None, mask=None)
            kw = {k: v for k, v in emb.to_dict().items() if v is not None}
            return gen.insert([ids], n, prompt_kwargs=[kw])[0]
        return gen, insert
    return tok, config.get("model_type", ""), make


def render(tok, text, model_type):
    kw = {"reasoning_effort": "none"} if model_type == "kolibri1" else {}
    enc = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                  add_generation_prompt=True, **kw)
    return list(enc["input_ids"] if hasattr(enc, "keys") else enc)


def generate(make, prompts, max_tokens, batch):
    """Greedy continuations for token-id prompts, in order.

    At most `batch` prompts are in the generator at a time; the next one goes
    in when one finishes, so long prompts do not all wait in memory.
    """
    gen, insert = make(max_tokens, batch)
    uids, out, done = [], {}, {}
    queue = list(prompts)
    try:
        while len(done) < len(prompts):
            while queue and len(uids) - len(done) < batch:
                uids.append(insert(queue.pop(0), max_tokens))
                out[uids[-1]] = []
            resp = gen.next()
            if isinstance(resp, tuple):
                resp = resp[-1]
            for r in resp:
                if r.finish_reason != "stop":
                    out[r.uid].append(r.token)
                if r.finish_reason is not None:
                    done[r.uid] = r.finish_reason
    finally:
        if hasattr(gen, "close"):
            gen.close()
    return [(out[u], done[u]) for u in uids]


def cmd_run(args):
    tok, model_type, make = load(args.model, args.loader)
    items = [json.loads(l) for l in open(DATA / "prompts.jsonl")]
    tasks = args.tasks.split(",")
    out_dir = DATA / "runs" / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        todo = [it for it in items if it["task"] == task]
        if args.limit:
            todo = todo[:: max(1, len(todo) // args.limit)]
        out = out_dir / f"{task}.jsonl"
        if out.exists():
            log(f"{args.arm} / {task}: done")
            continue
        t0 = time.time()
        groups = defaultdict(list)
        for it in todo:
            groups[it.get("length", 0)].append(it)
        rows = []
        for length, group in sorted(groups.items()):
            # KV memory grows with context: fewer sequences at a time for long prompts.
            batch = args.batch if not length else max(1, args.batch * 4096 // length)
            ids = [render(tok, it["prompt"], model_type) for it in group]
            res = generate(make, ids, MAX_TOKENS[task], batch)
            for it, p, (toks, fin) in zip(group, ids, res):
                rows.append({"id": it["id"], "prompt_tokens": len(p), "tokens": len(toks), "finish": fin,
                             "output": tok.decode(toks)})
            log(f"{args.arm} / {task}{f' @{length}' if length else ''}: {len(group)} prompts, "
                f"{time.time() - t0:.0f} s")
        tmp = out.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(out)


# ── report ───────────────────────────────────────────────────────────────────


def cmd_report(args):
    spec = importlib.util.spec_from_file_location("_mq", _HERE / "measure-quality.py")
    mq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mq)
    items = {it["id"]: it for it in map(json.loads, open(DATA / "prompts.jsonl"))}
    runs = {}
    for d in sorted((DATA / "runs").iterdir()):
        for f in d.glob("*.jsonl"):
            for r in map(json.loads, open(f)):
                it = items[r["id"]]
                ok, pred = SCORE[it["task"]](it, r["output"])
                runs.setdefault(d.name, {})[r["id"]] = {"ok": ok, "pred": pred, "finish": r["finish"],
                                                        "task": it["task"], "lang": it["lang"],
                                                        "group": it.get("length", ""), "checks": it.get("checks")}
    ref = runs.get(args.ref)
    result = {"date": time.strftime("%Y-%m-%d"), "prompts_sha256": (DATA / "prompts.sha256").read_text().strip(),
              "reference": args.ref, "arms": {}}
    print("| arm | task | group | n | score | 95% CI | Δ vs ref (95% CI) | right→wrong / wrong→right | cut off |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm, rows in runs.items():
        groups = defaultdict(list)
        for i, r in rows.items():
            for g in ("all", r["lang"], f"{r['group']}" if r["group"] else None):
                if g:
                    groups[(r["task"], g)].append(i)
        res = {}
        for (task, g), ids in sorted(groups.items()):
            ids = sorted(ids if ref is None or arm == args.ref else set(ids) & set(ref))
            k, n = sum(rows[i]["ok"] for i in ids), len(ids)
            e = {"n": n, "acc": k / n, "ci95": mq._wilson(k, n),
                 "cut_off": sum(rows[i]["finish"] == "length" for i in ids) / n}
            if task == "ifeval":
                per = [c == "1" for i in ids for c in rows[i]["pred"]]
                e["instruction_acc"] = sum(per) / len(per)
            if ref is not None and arm != args.ref:
                e.update(mq._paired([(ref[i]["ok"], ref[i]["pred"]) for i in ids],
                                    [(rows[i]["ok"], rows[i]["pred"]) for i in ids]))
            res[f"{task}/{g}"] = e
            lo, hi = e["ci95"]
            d = (f"{e['delta_acc'] * 100:+.1f} ({e['delta_ci95'][0] * 100:+.1f} to {e['delta_ci95'][1] * 100:+.1f})"
                 if "delta_acc" in e else "")
            fl = f"{e['ref_only']} / {e['arm_only']}" if "ref_only" in e else ""
            print(f"| {arm} | {task} | {g} | {n} | {e['acc']:.1%} | {lo:.1%}–{hi:.1%} | {d} | {fl} | "
                  f"{e['cut_off']:.0%} |")
        result["arms"][arm] = res
    out = DATA / f"generation-{result['date']}.json"
    out.write_text(json.dumps(result, indent=1))
    log(f"raw: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--lengths", default="8192,32768", help="RULER context lengths in tokens")
    p = sub.add_parser("run")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--loader", choices=["lm", "vlm"], default="vlm")
    p.add_argument("--tasks", default="mgsm,ifeval,ruler")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--limit", type=int, help="per task, spread over the prompts (smoke tests)")
    p = sub.add_parser("report")
    p.add_argument("--ref", required=True)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
