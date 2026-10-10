#!/usr/bin/env python3
"""Measure what quantization costs a model, against its original release.

Every arm -- the original as the reference, the shipped build and any control
build -- goes through the SAME layer-streamed forward pass. One decoder layer
is read from disk, run over all sequences of a set, and dropped before the
next one is read, so a reference that does not fit in memory (Kolibri's FP8
release: 73 GB) still runs on a 48 GB machine. What is kept is the final
normed hidden state; logits are computed from it at comparison time, with
each arm's own LM head.

    ./measure-quality.py --model kolibri prepare            # texts + benchmarks
    ./measure-quality.py --model kolibri check   --ckpt 3bit
    ./measure-quality.py --model kolibri forward --ckpt fp8 # all sets, resumable
    ./measure-quality.py --model kolibri report             # KL, PPL, accuracy

--model picks a profile from PROFILES: its checkpoints, its reference, the
build whose tokenizer and chat template make the sets, and its data
directory. measure-kolibri-quality.py and measure-apertus-quality.py are the
same with the profile fixed. --ckpt takes a name from the profile or a path.
Data, hidden states and results live outside the repository (--data): the
benchmark data is not ours to redistribute.

STOP THE SERVER FIRST. A streamed layer plus the hidden states of the
benchmark sets need ~10 GB; next to a 33 GB server that does not fit.

THE SETS.
  text         ~60k tokens of Wikipedia prose at pinned revisions, German and
               English, half of it from articles created after Kolibri's
               knowledge cutoff (2026-06-18), cut into 4096-token windows.
               Scored on every position: KL(ref || arm), top-1 agreement,
               perplexity; plus perplexity llama.cpp-style (second half of each
               window only), comparable to `llama-perplexity -c 4096`.
  belebele-*   Belebele deu_Latn / eng_Latn, 900 questions each.
  gmmlu-*      Global-MMLU-Lite de / en, 400 questions each.
               Multiple choice through the model's chat template with
               reasoning_effort=none; the answer is the letter with the highest
               logit among A-D at the first answer position. Accuracy with a
               95% Wilson interval; against the reference also the paired
               agreement and an exact McNemar test.

LOGITS ARE COMPUTED IN FP32 ON THE CPU. On the GPU an fp32 matmul deviates by
~1e-2 (measured during the port), the same order as the KL being measured.
"""

import argparse
import copy
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
PROFILES = {
    # FP8 release: dequantized lazily via the staging copy (see load_lazy).
    "kolibri": {
        "checkpoints": {
            "fp8": MODELS / "Kolibri-1-FP8",
            "3bit": MODELS / "Kolibri-1-MLX-3bit",
            "3bit-uniform": MODELS / "Kolibri-1-MLX-3bit-uniform",
        },
        "reference": "fp8",
        "tokenizer": "3bit",
        "data": "~/src/mlx/kolibri-quality",
    },
    # Needs the apertus1p5 branch of mlx-vlm: run with ~/src/mlx/.venv-apertus.
    # Only the language model is measured.
    "apertus": {
        "checkpoints": {
            "bf16": MODELS / "Apertus-v1.5-8B",
            "8bit": MODELS / "Apertus-v1.5-8B-MLX-8bit-omni",
            "4bit": MODELS / "Apertus-v1.5-8B-MLX-4bit",
        },
        "reference": "bf16",
        "tokenizer": "8bit",
        "data": "~/src/mlx/apertus-quality",
    },
}
# Set by use_profile().
CHECKPOINTS = {}
REFERENCE = None
TOKENIZER_ARM = None
DEFAULT_DATA = None
WINDOW = 4096
MIN_WINDOW = 512
LETTERS = "ABCD"
SAVE_EVERY = 10
UA = {"User-Agent": "local-sovereign-mlx/measure-quality"}

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

# Sets beyond Wikipedia (Unsloth: quants calibrated on Wikipedia look good on
# Wikipedia). Pinned to revisions; raw data stays in <data>/raw.
CALIB_V5 = (
    "https://gist.githubusercontent.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c/raw/"
    "571fda718462de863e5a0171078c175420c7649a/calibration_data_v5_rc.txt"
)
BELEBELE_REV = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
# The official EU languages as Belebele has them; Irish (gle) is not in Belebele.
EU_LANGS = (
    "bul_Cyrl ces_Latn dan_Latn deu_Latn ell_Grek eng_Latn est_Latn fin_Latn fra_Latn "
    "hrv_Latn hun_Latn ita_Latn lit_Latn lvs_Latn mlt_Latn nld_Latn pol_Latn por_Latn "
    "ron_Latn slk_Latn slv_Latn spa_Latn swe_Latn"
).split()
EU_PASSAGES = 20
# Multiple choice in less-represented EU languages (#19).
BELEBELE_EU = {"mlt": "mlt_Latn", "lvs": "lvs_Latn", "est": "est_Latn", "lit": "lit_Latn"}
OASST2 = ("OpenAssistant/oasst2", "179dd21fc55192153d94adb0e0ce8f69e222bf75",
          "2023-11-05_oasst2_ready.trees.jsonl.gz")
CHAT_LEN = 16384
CHAT_SEQS = 2  # per language
HERMES = ("NousResearch/hermes-function-calling-v1", "dae3e1d28cfbcf4b915c04ea1e072030529b4bda",
          "func-calling.json")
TOOL_CONVS = 40

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


def use_profile(name):
    global CHECKPOINTS, REFERENCE, TOKENIZER_ARM, DEFAULT_DATA
    prof = PROFILES[name]
    CHECKPOINTS = prof["checkpoints"]
    REFERENCE = prof["reference"]
    TOKENIZER_ARM = prof["tokenizer"]
    DEFAULT_DATA = prof["data"]


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
    """Build the sets named in --sets, or all that do not exist yet."""
    from transformers import AutoTokenizer

    data = args.data
    (data / "raw").mkdir(parents=True, exist_ok=True)
    (data / "sets").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(ckpt_path(TOKENIZER_ARM)))
    builders = {
        "core": (prepare_core, "text"),  # text, belebele-*, gmmlu-*
        "calib-v5": (prepare_calib_v5, "calib-v5"),
        "flores-eu": (prepare_flores_eu, "flores-eu"),
        "chat": (prepare_chat, "chat"),
        "tools": (prepare_tools, "tools"),
        "belebele-eu": (prepare_belebele_eu, "belebele-mlt"),
        "perm": (prepare_perm, "belebele-de-perm"),
    }
    names = args.sets.split(",") if args.sets else [
        n for n, (_, f) in builders.items() if not (data / "sets" / f"{f}.json").exists()
    ]
    for n in names:
        builders[n][0](data, tok)


def prepare_core(data, tok):
    letter_ids = _letter_ids(tok)

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

    # Belebele and Global-MMLU-Lite.
    for lang, fname in BELEBELE[1].items():
        _write_set(data, f"belebele-{lang}", "last", *_belebele_mc(tok, fname, lang),
                   {"letter_ids": letter_ids})
    for lang in ("de", "en"):
        _write_set(data, f"gmmlu-{lang}", "last", *_gmmlu_mc(data, tok, lang),
                   {"letter_ids": letter_ids})


def _letter_ids(tok):
    ids = [tok.encode(c, add_special_tokens=False) for c in LETTERS]
    if any(len(i) != 1 for i in ids) or len({i[0] for i in ids}) != 4:
        sys.exit(f"answer letters are not four distinct single tokens: {ids}")
    return ids


def _shift(options, answer, shift):
    """Options in a cyclic order: option j is shown at position (j + shift) % 4."""
    shown = [None] * 4
    for j, o in enumerate(options):
        shown[(j + shift) % 4] = o
    return shown, (answer + shift) % 4


def _belebele_mc(tok, fname, lang, shift=0, revision=None):
    """Belebele questions through the chat template; `lang` picks the prompt."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(BELEBELE[0], fname, repo_type="dataset", revision=revision)
    seqs, meta = [], []
    for line in open(path):
        q = json.loads(line)
        opts, ans = _shift([q[f"mc_answer{i}"] for i in range(1, 5)],
                           int(q["correct_answer_num"]) - 1, shift)
        user = PROMPT[lang].format(
            context=PASSAGE[lang].format(p=q["flores_passage"]), question=q["question"],
            a=opts[0], b=opts[1], c=opts[2], d=opts[3],
        )
        seqs.append(chat_ids(tok, user))
        m = {"id": f"{q['link']}#{q['question_number']}", "answer": LETTERS[ans]}
        if shift:
            m["shift"] = shift
        meta.append(m)
    return seqs, meta


def _gmmlu_mc(data, tok, lang, shift=0):
    raw = data / "raw" / f"gmmlu-lite-{lang}.json"
    if not raw.exists():
        raw.write_text(json.dumps(fetch_gmmlu(lang), ensure_ascii=False))
    seqs, meta = [], []
    for q in json.loads(raw.read_text()):
        opts, ans = _shift([q[f"option_{c}"] for c in "abcd"],
                           LETTERS.index(q["answer"].strip()), shift)
        user = PROMPT[lang].format(context="", question=q["question"],
                                   a=opts[0], b=opts[1], c=opts[2], d=opts[3])
        seqs.append(chat_ids(tok, user))
        m = {"id": q["sample_id"], "answer": LETTERS[ans],
             "cultural": q.get("cultural_sensitivity_label")}
        if shift:
            m["shift"] = shift
        meta.append(m)
    return seqs, meta


def prepare_belebele_eu(data, tok):
    """Belebele in less-represented EU languages. Passage, question and options
    in the language; the instruction in English, so no translation of ours."""
    for code, lang in BELEBELE_EU.items():
        _write_set(data, f"belebele-{code}", "last",
                   *_belebele_mc(tok, f"data/{lang}.jsonl", "en", revision=BELEBELE_REV),
                   {"letter_ids": _letter_ids(tok)})


def prepare_perm(data, tok):
    """The core multiple-choice sets with the options shifted by one position
    (A->B, B->C, C->D, D->A): position bias shows as answers that follow the
    letter instead of the content."""
    for lang, fname in BELEBELE[1].items():
        _write_set(data, f"belebele-{lang}-perm", "last",
                   *_belebele_mc(tok, fname, lang, shift=1, revision=BELEBELE_REV),
                   {"letter_ids": _letter_ids(tok)})
    for lang in ("de", "en"):
        _write_set(data, f"gmmlu-{lang}-perm", "last", *_gmmlu_mc(data, tok, lang, shift=1),
                   {"letter_ids": _letter_ids(tok)})


def _fetch_raw(data, name, url):
    raw = data / "raw" / name
    if not raw.exists():
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw.write_bytes(resp.read())
    return raw


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _windows(ids):
    return [ids[i : i + WINDOW] for i in range(0, len(ids), WINDOW)
            if len(ids[i : i + WINDOW]) >= MIN_WINDOW]


def prepare_calib_v5(data, tok):
    """Calibration v5 (bartowski / tristandruyen), the set Unsloth uses for fair KLD tests."""
    raw = _fetch_raw(data, "calibration_data_v5_rc.txt", CALIB_V5)
    seqs = _windows(tok.encode(raw.read_text(), add_special_tokens=False))
    _write_set(data, "calib-v5", "all", seqs, [{"lang": "mixed"} for _ in seqs],
               {"sources": [{"url": CALIB_V5, "sha256": _sha256(raw)}]})


def prepare_flores_eu(data, tok):
    """FLORES passages in the EU languages, via Belebele (FLORES+ itself is gated).

    The first EU_PASSAGES distinct passages of each language file, in file
    order, joined into one text per language.
    """
    from huggingface_hub import hf_hub_download

    seqs, meta, sources = [], [], []
    for lang in EU_LANGS:
        path = hf_hub_download(BELEBELE[0], f"data/{lang}.jsonl", repo_type="dataset",
                               revision=BELEBELE_REV)
        passages = []
        for line in open(path):
            p = json.loads(line)["flores_passage"]
            if p not in passages:
                passages.append(p)
            if len(passages) == EU_PASSAGES:
                break
        ids = tok.encode("\n\n".join(passages), add_special_tokens=False)
        for w in _windows(ids) or [ids]:
            seqs.append(w)
            meta.append({"lang": lang})
        sources.append({"lang": lang, "file": f"data/{lang}.jsonl", "sha256": _sha256(path)})
    _write_set(data, "flores-eu", "all", seqs, meta,
               {"sources": [{"repo": BELEBELE[0], "revision": BELEBELE_REV}] + sources})


def _render(tok, msgs, tools=None, gen=False):
    enc = tok.apply_chat_template(msgs, tools=tools, tokenize=True, add_generation_prompt=gen)
    return list(enc["input_ids"] if hasattr(enc, "keys") else enc)


def _common(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def _assistant_spans(tok, msgs, tools, full):
    """[start, end) token ranges of the assistant turns in the rendered conversation.

    Text turns are found by their content in the rendered string (templates
    may render earlier turns differently from the last one, e.g. Kolibri drops
    the empty think block before the last user turn); turns that are only tool
    calls by the token prefix of the generation prompt and the turn itself.
    """
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(enc["input_ids"]) != full:
        sys.exit("rendered text does not re-tokenize to the template's token ids")
    offsets = enc["offset_mapping"]
    spans, cursor = [], 0
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        content = (m.get("content") or "").strip()
        at = text.find(content, cursor) if content else -1
        if at >= 0:
            c0, c1 = at, at + len(content)
            cursor = c1
            toks = [t for t, (o0, o1) in enumerate(offsets) if o1 > c0 and o0 < c1]
            spans.append([toks[0], toks[-1] + 1])
        else:
            start = _common(_render(tok, msgs[:i], tools, gen=True), full)
            end = _common(_render(tok, msgs[: i + 1], tools), full)
            spans.append([start, end])
    return spans


def _oasst_threads(path, lang):
    """Best-ranked path through each conversation tree of one language."""
    import gzip

    threads = []
    for line in gzip.open(path, "rt"):
        tree = json.loads(line)
        node = tree["prompt"]
        if node.get("lang") != lang:
            continue
        msgs = []
        while node:
            role = "user" if node["role"] == "prompter" else "assistant"
            msgs.append({"role": role, "content": node["text"]})
            replies = node.get("replies") or []
            node = min(replies, key=lambda r: (r.get("rank") is None, r.get("rank") or 0),
                       default=None)
        if msgs[-1]["role"] == "user":
            msgs.pop()
        if len(msgs) >= 2:
            threads.append((tree["message_tree_id"], msgs))
    threads.sort()
    random.Random(0).shuffle(threads)
    return threads


def prepare_chat(data, tok):
    """Long multi-turn chats (oasst2, de / en): threads chained into one dialogue
    each, up to CHAT_LEN tokens through the chat template."""
    from huggingface_hub import hf_hub_download

    repo, rev, fname = OASST2
    path = hf_hub_download(repo, fname, repo_type="dataset", revision=rev)
    seqs, meta, spans = [], [], []
    for lang in ("de", "en"):
        threads = iter(_oasst_threads(path, lang))
        for _ in range(CHAT_SEQS):
            msgs, ids, used = [], [], []
            for tid, thread in threads:
                cand = _render(tok, msgs + thread)
                if len(cand) > CHAT_LEN:
                    if msgs:
                        break
                    continue
                msgs, ids = msgs + thread, cand
                used.append(tid)
            seqs.append(ids)
            spans.append(_assistant_spans(tok, msgs, None, ids))
            meta.append({"lang": lang, "threads": used})
    _write_set(data, "chat", "all", seqs, meta,
               {"spans": spans, "sources": [{"repo": repo, "revision": rev, "file": fname,
                                             "sha256": _sha256(path)}]})


def _hermes_messages(conv):
    """hermes-function-calling turns as chat messages with structured tool calls."""
    import re

    msgs = []
    for turn in conv:
        role, text = turn["from"], turn["value"]
        if role == "system":
            continue
        if role == "human":
            msgs.append({"role": "user", "content": text})
        elif role == "gpt":
            calls = [json.loads(c) for c in
                     re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)]
            content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()
            m = {"role": "assistant", "content": content}
            if calls:
                m["tool_calls"] = [{"type": "function", "function": {
                    "name": c["name"], "arguments": c.get("arguments", {})}} for c in calls]
            msgs.append(m)
        elif role == "tool":
            for r in re.findall(r"<tool_response>\s*(.*?)\s*</tool_response>", text, re.S):
                msgs.append({"role": "tool", "content": r})
    return msgs


def prepare_tools(data, tok):
    """Tool-calling conversations (hermes-function-calling-v1, multi-turn), one per
    sequence, with the tools passed to the chat template."""
    from huggingface_hub import hf_hub_download

    repo, rev, fname = HERMES
    path = hf_hub_download(repo, fname, repo_type="dataset", revision=rev)
    rows = json.load(open(path))
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    seqs, meta, spans = [], [], []
    for i in order:
        try:
            tools = json.loads(rows[i]["tools"])
            msgs = _hermes_messages(rows[i]["conversations"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if not any(m.get("tool_calls") for m in msgs) or msgs[-1]["role"] != "assistant":
            continue
        ids = _render(tok, msgs, tools)
        seqs.append(ids)
        spans.append(_assistant_spans(tok, msgs, tools, ids))
        meta.append({"lang": "en", "id": rows[i]["id"]})
        if len(seqs) == TOOL_CONVS:
            break
    _write_set(data, "tools", "all", seqs, meta,
               {"spans": spans, "sources": [{"repo": repo, "revision": rev, "file": fname,
                                             "sha256": _sha256(path)}]})


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


# Text-only builds in mlx-lm's format (e.g. community Apertus 1.5 "text" builds,
# model_type "apertus") load through mlx-lm; everything else through mlx-vlm.
MLX_LM_TYPES = {"apertus"}


def load_lazy(path):
    """The model with lazy weights; the FP8 release goes through the staging copy."""
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") in MLX_LM_TYPES:
        from mlx_lm.utils import load_model

        model, _ = load_model(path, lazy=True)
        return model
    from mlx_vlm import load

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


def streamed_forward(lm, seqs, keep_last, partial=None, chunk=None):
    """Final normed hidden states, one decoder layer in memory at a time.

    Returns a list with one array per sequence: [L, H] or, with keep_last, [H].
    `partial` is a path for resume state, written every SAVE_EVERY layers.
    With `chunk`, each sequence runs through a layer in pieces of that many
    tokens against the layer's KV cache, the way a server prefills. Same
    weights and math, a different order of floating-point operations: the
    reference run this way gives the numerical noise floor.
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

    # Fresh caches are copied from these; make_cache needs the layers, which
    # are dropped one by one below.
    protos = lm.make_cache() if chunk else None
    masks = {}
    for i in range(start, len(layers)):
        t0 = time.time()
        layer = layers[i]
        sliding = getattr(layer, "use_sliding", False)
        window = inner.sliding_window if sliding else None
        for j, h in enumerate(hs):
            L = h.shape[1]
            if chunk:
                cache = copy.deepcopy(protos[i])
                parts = []
                for a in range(0, L, chunk):
                    hc = h[:, a : a + chunk]
                    parts.append(layer(hc, create_attention_mask(hc, cache, window_size=window),
                                       cache))
                    mx.eval(parts[-1])
                hs[j] = mx.concatenate(parts, axis=1)
                mx.eval(hs[j])
                continue
            key = (L, sliding)
            if key not in masks:
                masks[key] = create_attention_mask(h, None, window_size=window)
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
    head_weight(path, head_path(args.data, arm))
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
            lm, st["seqs"], st["score"] == "last", out.with_suffix(".partial.safetensors"),
            chunk=args.chunk,
        )
        if st["score"] == "last":
            mx.save_safetensors(str(out), {"h": mx.stack(hs)})
        else:
            mx.save_safetensors(str(out), {"h": mx.concatenate(hs, axis=0)})
        out.with_suffix(".partial.safetensors").unlink(missing_ok=True)
        del lm, hs
        mx.clear_cache()


# ── heads and scoring ────────────────────────────────────────────────────────


def head_weight(path, cache=None):
    """The arm's LM head as an fp32 matrix [V, H], dequantized if quantized.

    With `cache` (a .safetensors path) the head's stored tensors are kept next
    to the hidden states, so a report no longer needs the checkpoint.
    """
    import mlx.core as mx

    if cache is not None and cache.exists():
        t, meta = mx.load(str(cache), return_metadata=True)
    else:
        head = load_lazy(path).lm_head
        t = {"weight": head.weight}
        meta = {}
        if hasattr(head, "scales"):
            t["scales"] = head.scales
            if getattr(head, "biases", None) is not None:
                t["biases"] = head.biases
            meta = {"group_size": str(head.group_size), "bits": str(head.bits),
                    "mode": getattr(head, "mode", "affine")}
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(str(cache), t, metadata=meta)
    if "scales" in t:
        w = mx.dequantize(t["weight"], t["scales"], t.get("biases"),
                          group_size=int(meta["group_size"]), bits=int(meta["bits"]),
                          mode=meta["mode"])
    else:
        w = t["weight"]
    w = w.astype(mx.float32)
    mx.eval(w)
    return w


def head_path(data, arm):
    return data / "hidden" / arm / "lm_head.safetensors"


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


EQUIV_MARGIN = 0.01  # TOST margin: ±1 percentage point, fixed before the runs


def _paired_diff_ci(a, b, c, d, z):
    """Newcombe's hybrid score interval (method 10) for a paired difference.

    Checked against Newcombe (1998), table on p. 2641, second row (20/12/2/16):
    0.0562 to 0.3292, as StatsDirect and NCSS PASS report it.

    a: both right, b: reference only, c: arm only, d: both wrong. Returns the
    interval for acc(arm) - acc(ref) = (c - b) / n.
    """
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = _wilson(a + b, n, z)
    l2, u2 = _wilson(a + c, n, z)
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    num = a * d - b * c
    if num > 0:
        num = max(num - n / 2, 0.0)
    phi = num / den if den else 0.0
    diff = p2 - p1
    dl = math.sqrt(max(0.0, (p2 - l2) ** 2 - 2 * phi * (p2 - l2) * (u1 - p1) + (u1 - p1) ** 2))
    du = math.sqrt(max(0.0, (u2 - p2) ** 2 - 2 * phi * (u2 - p2) * (p1 - l1) + (p1 - l1) ** 2))
    return (diff - dl, diff + du)


def _paired(ref, arm):
    """Paired comparison of one arm with the reference on the same questions.

    ref, arm: lists of (correct, predicted letter). Flips (Dutta et al. 2024)
    are right <-> wrong changes, split by direction; same_answer also counts
    wrong -> other wrong. Equivalence by TOST: the 90% interval of the paired
    difference (two one-sided tests at 5%) lies inside ±EQUIV_MARGIN.
    """
    n = len(ref)
    a = sum(r and x for (r, _), (x, _) in zip(ref, arm))
    b = sum(r and not x for (r, _), (x, _) in zip(ref, arm))
    c = sum(x and not r for (r, _), (x, _) in zip(ref, arm))
    d = n - a - b - c
    ci90 = _paired_diff_ci(a, b, c, d, 1.6449)
    return {
        "same_answer": sum(p == q for (_, p), (_, q) in zip(ref, arm)) / n,
        "delta_acc": (c - b) / n,
        "delta_ci95": _paired_diff_ci(a, b, c, d, 1.96),
        "delta_ci90": ci90,
        "equivalent": -EQUIV_MARGIN < ci90[0] and ci90[1] < EQUIV_MARGIN,
        "mcnemar_p": _mcnemar(b, c),
        "ref_only": b, "arm_only": c,
        "flips": b + c, "flip_rate": (b + c) / n,
    }


def text_scores(h_ref, h_arm, w_ref, w_arm, seqs, meta, spans=None, chunk=512):
    """Per scored position: KL(ref||arm), top-1 agreement, NLL of both, group keys.

    Row: (lang, period, second half, KL, NLL ref, NLL arm, same top, target in
    an assistant span).
    """
    import mlx.core as mx

    rows = []
    off = 0
    with mx.stream(mx.cpu):
        for j, (s, m) in enumerate(zip(seqs, meta)):
            inside = set()
            for a, b in (spans[j] if spans else []):
                inside.update(range(a, b))
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
                    rows.append((m["lang"], m.get("period"), a + i >= L // 2, k, r, q, g,
                                 a + i + 1 in inside))
            off += L
    return rows


def _summ(rows):
    """KL distribution as llama-perplexity --kl-divergence reports it (Unsloth's numbers).

    Percentiles by nearest rank on the sorted per-token values; p99.9 is the
    worst token in a thousand, max the single worst. top1_agree is llama.cpp's
    "same top p".
    """
    import statistics

    kls = sorted(r[3] for r in rows)
    n = len(kls)

    def pct(q):
        return kls[min(n - 1, int(q * n))]

    return {
        "n": n,
        "kl_mean": sum(kls) / n,
        "kl_median": statistics.median(kls),
        "kl_p90": pct(0.90),
        "kl_p99": pct(0.99),
        "kl_p999": pct(0.999),
        "kl_max": kls[-1],
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
        sys.exit(f"no reference hidden states; run `forward --ckpt {REFERENCE}` first")
    sets = load_sets(data)
    heads = {a: head_weight(ckpt_path(a), head_path(data, a)) for a in arms}
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

    result["kld"] = {}
    for name, st in sets.items():
        if st["score"] != "all" or name == "text":
            continue
        result["kld"][name] = {}
        for arm in arms:
            if arm == REFERENCE or not hidden_path(data, arm, name).exists():
                continue
            h_ref = mx.load(str(hidden_path(data, REFERENCE, name)))["h"]
            h_arm = mx.load(str(hidden_path(data, arm, name)))["h"]
            rows = text_scores(h_ref, h_arm, heads[REFERENCE], heads[arm], st["seqs"],
                               st["meta"], st.get("spans"))
            groups = {"all": rows}
            for lang in sorted({m["lang"] for m in st["meta"]}):
                groups[lang] = [r for r in rows if r[0] == lang]
            groups["second-half"] = [r for r in rows if r[2]]
            if st.get("spans"):
                groups["assistant"] = [r for r in rows if r[7]]
            result["kld"][name][arm] = {g: _summ(rs) for g, rs in groups.items() if rs}

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
                entry.update(_paired(list(zip(ref["correct"], ref["preds"])),
                                     list(zip(sc["correct"], sc["preds"]))))
            result["mc"][name][arm] = entry
        # Subgroups stored per question, e.g. Global-MMLU-Lite's cultural label.
        labels = sorted({m["cultural"] for m in st["meta"] if m.get("cultural")})
        for label in labels:
            idx = [i for i, m in enumerate(st["meta"]) if m.get("cultural") == label]
            grp = result.setdefault("mc_groups", {}).setdefault(f"{name}/{label}", {})
            for arm, sc in per_arm.items():
                k = sum(sc["correct"][i] for i in idx)
                entry = {"n": len(idx), "acc": k / len(idx), "ci95": _wilson(k, len(idx))}
                if ref is not None and arm != REFERENCE:
                    entry.update(_paired([(ref["correct"][i], ref["preds"][i]) for i in idx],
                                         [(sc["correct"][i], sc["preds"][i]) for i in idx]))
                grp[arm] = entry

    # Position bias (#20): the same questions with the options shifted. An
    # answer that follows the content picks the same option in both orders.
    result["position"] = {}
    for name, st in sets.items():
        base = name[: -len("-perm")]
        if not name.endswith("-perm") or base not in sets:
            continue
        shift = st["meta"][0]["shift"]
        res = {}
        for arm in arms:
            p0, p1 = hidden_path(data, arm, base), hidden_path(data, arm, name)
            if not (p0.exists() and p1.exists()):
                continue
            s0 = mc_scores(mx.load(str(p0))["h"], heads[arm], st["letter_ids"], sets[base]["meta"])
            s1 = mc_scores(mx.load(str(p1))["h"], heads[arm], st["letter_ids"], st["meta"])
            back = [LETTERS[(LETTERS.index(x) - shift) % 4] for x in s1["preds"]]
            n = len(back)
            res[arm] = {
                "n": n,
                "acc": sum(s0["correct"]) / n,
                "acc_shifted": sum(s1["correct"]) / n,
                "same_option": sum(a == b for a, b in zip(s0["preds"], back)) / n,
                "letters": {c: s0["preds"].count(c) / n for c in LETTERS},
                "letters_shifted": {c: s1["preds"].count(c) / n for c in LETTERS},
                "answers": {c: sum(m["answer"] == c for m in sets[base]["meta"]) / n
                            for c in LETTERS},
            }
        result["position"][base] = res

    out = data / "results" / f"quality-{result['date']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    with open(out.with_name(f"per-question-{result['date']}.jsonl"), "w") as fh:
        for row in per_question:
            fh.write(json.dumps(row) + "\n")
    _print_report(result)
    _print_position(result)
    log(f"raw: {out}")


def _versions():
    import importlib.metadata as md
    import subprocess

    commit = subprocess.run(
        ["git", "-C", str(_HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return {"mlx": md.version("mlx"), "mlx_vlm": md.version("mlx-vlm"), "repo_commit": commit}


def _print_report(r):
    print(f"\n## KL divergence and perplexity vs {REFERENCE} (text set)\n")
    print("| arm | group | tokens | mean KL | median | p90 | p99 | p99.9 | max | same top | "
          "PPL ref | PPL arm |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm, groups in r["text"].items():
        for g, s in groups.items():
            print(f"| {arm} | {g} | {s['n']:,} | {s['kl_mean']:.4f} | {s['kl_median']:.4f} | "
                  f"{s['kl_p90']:.3f} | {s['kl_p99']:.3f} | {s['kl_p999']:.3f} | {s['kl_max']:.2f} | "
                  f"{s['top1_agree']:.1%} | {s['ppl_ref']:.3f} | {s['ppl_arm']:.3f} |")
    for name, arms in r.get("kld", {}).items():
        print(f"\n## KL divergence vs {REFERENCE} ({name})\n")
        print("| arm | group | tokens | mean KL | median | p90 | p99 | p99.9 | max | same top | "
              "PPL ref | PPL arm |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for arm, groups in arms.items():
            for g, s in groups.items():
                print(f"| {arm} | {g} | {s['n']:,} | {s['kl_mean']:.4f} | {s['kl_median']:.4f} | "
                      f"{s['kl_p90']:.3f} | {s['kl_p99']:.3f} | {s['kl_p999']:.3f} | "
                      f"{s['kl_max']:.2f} | {s['top1_agree']:.1%} | {s['ppl_ref']:.3f} | "
                      f"{s['ppl_arm']:.3f} |")
    print("\n## Multiple choice (reasoning_effort=none, letter logits)\n")
    print(f"Δ vs {REFERENCE} with the paired 95% interval (Newcombe); equivalent = TOST, "
          f"90% interval inside ±{EQUIV_MARGIN:.0%}.\n")
    print("| set | arm | n | accuracy | Δ | Δ 95% CI | equivalent | right→wrong | wrong→right | "
          "McNemar p | same answer | letter top-1 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    groups = list(r["mc"].items()) + list(r.get("mc_groups", {}).items())
    for name, arms in groups:
        for arm, e in arms.items():
            if "delta_acc" not in e:
                print(f"| {name} | {arm} | {e['n']} | {e['acc']:.1%} | | | | | | | | "
                      f"{e['letter_top1']:.1%} |" if "letter_top1" in e else
                      f"| {name} | {arm} | {e['n']} | {e['acc']:.1%} | | | | | | | | |")
                continue
            lo, hi = e["delta_ci95"]
            top1 = f"{e['letter_top1']:.1%}" if "letter_top1" in e else ""
            print(f"| {name} | {arm} | {e['n']} | {e['acc']:.1%} | {e['delta_acc']:+.1%} | "
                  f"{lo:+.1%} to {hi:+.1%} | {'yes' if e['equivalent'] else 'no'} | "
                  f"{e['ref_only']} | {e['arm_only']} | {e['mcnemar_p']:.3f} | "
                  f"{e['same_answer']:.1%} | {top1} |")


def _print_position(r):
    if not r.get("position"):
        return
    print("\n## Position bias: options shifted by one (A→B … D→A)\n")
    print("| set | arm | accuracy | shifted | same option chosen | predicted A/B/C/D | "
          "correct A/B/C/D |")
    print("|---|---|---|---|---|---|---|")
    for base, arms in r["position"].items():
        for arm, e in arms.items():
            pred = "/".join(f"{e['letters'][c]:.0%}" for c in LETTERS)
            ans = "/".join(f"{e['answers'][c]:.0%}" for c in LETTERS)
            print(f"| {base} | {arm} | {e['acc']:.1%} | {e['acc_shifted']:.1%} | "
                  f"{e['same_option']:.1%} | {pred} | {ans} |")


def main(profile=None, doc=__doc__):
    ap = argparse.ArgumentParser(description=doc.split("\n\n")[0])
    if profile is None:
        ap.add_argument("--model", required=True, choices=sorted(PROFILES))
    ap.add_argument("--data", type=Path, help="data directory (default: the profile's)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--sets", help="core,calib-v5,flores-eu,chat,tools,belebele-eu,perm "
                   "(default: all not yet built)")
    p = sub.add_parser("check")
    p.add_argument("--ckpt", help="default: the profile's tokenizer build")
    p = sub.add_parser("forward")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--arm", help="name of the arm (default: --ckpt)")
    p.add_argument("--sets", help="comma-separated subset of sets")
    p.add_argument("--chunk", type=int,
                   help="prefill in pieces of this many tokens (noise-floor arm, e.g. 512)")
    sub.add_parser("report")
    args = ap.parse_args()
    use_profile(profile or args.model)
    args.data = (args.data or Path(DEFAULT_DATA)).expanduser()
    if args.cmd == "check" and not args.ckpt:
        args.ckpt = TOKENIZER_ARM
    {"prepare": cmd_prepare, "check": cmd_check, "forward": cmd_forward, "report": cmd_report}[
        args.cmd
    ](args)


if __name__ == "__main__":
    main()
