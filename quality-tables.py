#!/usr/bin/env python3
"""Markdown tables for docs/*-quality/README.md from a measure-quality.py result.

    ./quality-tables.py docs/apertus-quality/quality-2026-10-09.json --ref bf16 \
        --arms bf16-chunked,8bit,4bit --names "noise floor,8-bit (shipped),4-bit RTN" \
        --mc-arms 8bit,4bit

Prints the tables; the README text around them is written by hand.
"""

import argparse
import json

SET_NAMES = {
    "text": "Wikipedia de/en",
    "calib-v5": "Calibration v5",
    "chat": "chat (oasst2)",
    "tools": "tool calling",
    "flores-eu": "FLORES, 23 EU languages",
}
MC_NAMES = {
    "belebele-de": "Belebele de (900)",
    "belebele-en": "Belebele en (900)",
    "gmmlu-de": "Global-MMLU-Lite de (400)",
    "gmmlu-en": "Global-MMLU-Lite en (400)",
    "gmmlu-de/CS": "… de, culturally sensitive (200)",
    "gmmlu-de/CA": "… de, culturally agnostic (200)",
    "gmmlu-en/CS": "… en, culturally sensitive (200)",
    "gmmlu-en/CA": "… en, culturally agnostic (200)",
    "belebele-mlt": "Belebele Maltese (900)",
    "belebele-lvs": "Belebele Latvian (900)",
    "belebele-est": "Belebele Estonian (900)",
    "belebele-lit": "Belebele Lithuanian (900)",
    "belebele-de-perm": "Belebele de, options shifted (900)",
    "belebele-en-perm": "Belebele en, options shifted (900)",
    "gmmlu-de-perm": "Global-MMLU-Lite de, options shifted (400)",
    "gmmlu-en-perm": "Global-MMLU-Lite en, options shifted (400)",
}


def pp(x):
    return f"{x * 100:+.1f}".replace("-", "−")


def kld_table(r, arms, names):
    sets = {"text": r["text"], **r.get("kld", {})}
    print("| set | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(arms))
    for key, label in SET_NAMES.items():
        if key not in sets:
            continue
        cells = []
        for arm in arms:
            g = sets[key].get(arm, {}).get("all")
            cells.append(f"{g['kl_mean']:.4f} / {g['kl_p999']:.3f} / {g['top1_agree']:.1%}"
                         if g else "–")
        print(f"| {label} | " + " | ".join(cells) + " |")
    for key in ("chat", "tools"):
        if key in sets:
            cells = []
            for arm in arms:
                g = sets[key].get(arm, {}).get("assistant")
                cells.append(f"{g['kl_mean']:.4f} / {g['kl_p999']:.3f} / {g['top1_agree']:.1%}"
                             if g else "–")
            print(f"| {SET_NAMES[key]}, assistant turns | " + " | ".join(cells) + " |")


def kld_full(r, arm):
    sets = {"text": r["text"], **r.get("kld", {})}
    print("| set | tokens | mean | median | p90 | p99 | p99.9 | max | same top | PPL ref → arm |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key, label in SET_NAMES.items():
        g = sets.get(key, {}).get(arm, {}).get("all")
        if g:
            print(f"| {label} | {g['n']:,} | {g['kl_mean']:.4f} | {g['kl_median']:.4f} | "
                  f"{g['kl_p90']:.3f} | {g['kl_p99']:.3f} | {g['kl_p999']:.3f} | "
                  f"{g['kl_max']:.2f} | {g['top1_agree']:.1%} | "
                  f"{g['ppl_ref']:.2f} → {g['ppl_arm']:.2f} |")


def eu_table(r, arms, names):
    eu = r.get("kld", {}).get("flores-eu", {})
    langs = sorted(k for k in eu.get(arms[-1], {}) if "_" in k)
    print("| language | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(arms))
    for lang in langs:
        cells = []
        for arm in arms:
            g = eu.get(arm, {}).get(lang)
            cells.append(f"{g['kl_mean']:.4f} / {g['top1_agree']:.1%}" if g else "–")
        print(f"| {lang} | " + " | ".join(cells) + " |")


def mc_table(r, ref, arms, names):
    groups = {**r["mc"], **r.get("mc_groups", {})}
    head = " | ".join(f"{n} | Δ (95% CI) | equiv. | flips −/+" for n in names)
    print(f"| set | {ref} | {head} |")
    print("|---|---|" + "---|---|---|---|" * len(arms))
    for key, label in MC_NAMES.items():
        if key not in groups:
            continue
        g = groups[key]
        cells = [f"{g[ref]['acc']:.1%}"]
        for arm in arms:
            e = g.get(arm)
            if not e:
                cells += ["–"] * 4
                continue
            lo, hi = e["delta_ci95"]
            cells += [f"{e['acc']:.1%}", f"{pp(e['delta_acc'])} ({pp(lo)} to {pp(hi)})",
                      "yes" if e["equivalent"] else "no", f"{e['ref_only']}/{e['arm_only']}"]
        print(f"| {label} | " + " | ".join(cells) + " |")


def position_table(r, ref, arms, names):
    pos = r.get("position", {})
    print("| set | arm | accuracy | options shifted | same option chosen | predicted A/B/C/D |")
    print("|---|---|---|---|---|---|")
    for base, res in pos.items():
        for arm, name in [(ref, ref)] + list(zip(arms, names)):
            e = res.get(arm)
            if e:
                pred = " / ".join(f"{e['letters'][c]:.0%}" for c in "ABCD")
                print(f"| {MC_NAMES.get(base, base)} | {name} | {e['acc']:.1%} | "
                      f"{e['acc_shifted']:.1%} | {e['same_option']:.1%} | {pred} |")
        first = next(iter(res.values()), None)
        if first:
            ans = " / ".join(f"{first['answers'][c]:.0%}" for c in "ABCD")
            print(f"| {MC_NAMES.get(base, base)} | correct answers | | | | {ans} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--mc-arms", help="arms for the multiple-choice table (default: --arms)")
    args = ap.parse_args()
    r = json.load(open(args.result))
    arms, names = args.arms.split(","), args.names.split(",")
    print("### KL divergence: mean / p99.9 / same top\n")
    kld_table(r, arms, names)
    for arm, name in zip(arms, names):
        print(f"\n### {name}: full KL distribution\n")
        kld_full(r, arm)
    print("\n### EU languages: mean KL / same top\n")
    eu_table(r, arms, names)
    print("\n### Multiple choice\n")
    mc_arms = args.mc_arms.split(",") if args.mc_arms else arms
    mc_names = [names[arms.index(a)] for a in mc_arms]
    mc_table(r, args.ref, mc_arms, mc_names)
    print("\n### Position bias\n")
    position_table(r, args.ref, mc_arms, mc_names)


if __name__ == "__main__":
    main()
