#!/usr/bin/env python3
"""One line per measured build for the sovereign-models verified label (sovereign-models#23).

Reads a measure-quality.py result and writes, for the shipped build: the mean
KL divergence and the noise floor per text set, the worst ratio, and whether
any multiple-choice set loses significantly (paired 95% interval entirely
below zero). sovereign-models applies the thresholds.

    ./quality-summary.py <result.json> --build <hf repo> --arm <arm> --ref <ref> --noise <noise arm>
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--build", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--noise", required=True)
    a = ap.parse_args()
    r = json.load(open(a.result))
    sets = {"text": r["text"], **r.get("kld", {})}
    kl = {s: v[a.arm]["all"]["kl_mean"] for s, v in sets.items() if a.arm in v}
    noise = {s: v[a.noise]["all"]["kl_mean"] for s, v in sets.items() if a.noise in v}
    losses = [k for k, v in {**r["mc"], **r.get("mc_groups", {})}.items()
              if a.arm in v and v[a.arm].get("delta_ci95", [0, 0])[1] < 0]
    print(json.dumps({
        "build": a.build, "measured": r["date"], "reference": a.ref,
        "kl_mean_max": max(kl.values()),
        "kl_noise_ratio_max": max(kl[s] / noise[s] for s in kl if noise.get(s)),
        "kl_by_set": kl, "noise_by_set": noise,
        "mc_significant_losses": losses,
    }))


if __name__ == "__main__":
    main()
