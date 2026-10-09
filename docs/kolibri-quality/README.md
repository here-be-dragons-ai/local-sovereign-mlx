# Kolibri-1-MLX-3bit: quality loss and time to first token

Measurements for issue #8, 2026-10-08, Apple M5 Pro, 48 GB,
`iogpu.wired_limit_mb=40960`, mlx 0.32.2, mlx-vlm 0.7.4 (repo commit `4f68407`
plus the two measurement scripts).

| file | content |
|---|---|
| `quality-2026-10-08.json` | KL / perplexity per arm and group, multiple-choice accuracy and flips, text sources (report re-run 2026-10-09 with p90 / p99.9 / max and flips; earlier values unchanged) |
| `per-question-2026-10-08.jsonl` | per question and arm: set, id, predicted letter, answer, correct |
| `ttft-2026-10-08.jsonl` | one line per TTFT request, with server flags and versions |
| `ttft-reserve-2026-10-08.jsonl` | the same with `APC_MEMORY_RESERVE_GB=1.5`, one run up to 64k |

## Quality

**Method.** Three arms go through the same layer-streamed forward pass
(`measure-kolibri-quality.py`): the FP8 release (`Aleph-Alpha/Kolibri-1`) as
the reference, the shipped 3/6-bit build, and a uniform 3-bit control
(`convert-kolibri.py --other-bits 3`). The streamed pass is bit-identical to
the ordinary in-memory forward on the 3-bit checkpoint (`check`: max |Δh| = 0).
Logits are computed in fp32 on the CPU from the final hidden state and each
arm's own LM head.

**KL divergence and perplexity.** 75,900 scored tokens of Wikipedia prose at
pinned revisions (`sources` in the JSON), German and English, 4096-token
windows. "post" articles were created after Kolibri's knowledge cutoff
(2026-06-18).

| arm | bits/weight | mean KL | median | p90 | p99 | p99.9 | max | same top | PPL (FP8: 16.98) |
|---|---|---|---|---|---|---|---|---|---|
| 3/6-bit (shipped) | 3.61 | 0.114 | 0.021 | 0.133 | 2.07 | 10.3 | 19.8 | 88.9% | 17.11 |
| uniform 3-bit | 3.51 | 0.376 | 0.138 | 0.694 | 5.42 | 13.1 | 23.5 | 76.5% | 19.38 |

The columns follow `llama-perplexity --kl-divergence`, the numbers Unsloth
reports for its GGUFs (percentiles by nearest rank; "same top" is top-1
agreement). The tail is heavy: one token in a thousand has a KL above 10
nats in the shipped build: there its next-token distribution differs
drastically from FP8's. The mean hides this. For scale, Unsloth's 4-bit GGUFs
of Qwen3.5 / Qwen3.8 report p99.9 between about 0.4 and 0.8, at ~4.5 bits per
weight and on their own text.

By language, the shipped build: German mean KL 0.106, English 0.121; before /
after the cutoff 0.110 / 0.101 (de) and 0.092 / 0.155 (en).

Cross-runtime: `llama-perplexity -c 4096` on the same text gives **17.75 ±
0.30** for the Q3_K_S GGUF (`Eliasfpv28/Kolibri-1-Q3_K_S-GGUF`, 31.5 GiB). In the same convention -- second half of each window scored -- the
FP8 reference is 15.69, the shipped 3/6-bit build 15.75, uniform 3-bit 17.63.
The windows are not cut identically (llama.cpp splits across article
boundaries), so the numbers are comparable, not equal. No KL for the GGUF: that
needs llama.cpp's logit file format for the reference.

**Multiple choice.** Kolibri's chat template with `reasoning_effort=none`,
options A-D in the user turn, scored by the letter logits at the first answer
position. Δ and McNemar (exact, two-sided) are paired against FP8.

| set | FP8 | 3/6-bit (shipped) | Δ | flips | p | uniform 3-bit | Δ | flips | p |
|---|---|---|---|---|---|---|---|---|---|
| Belebele de (900) | 92.9% | 93.1% | +0.2 pp | 1.8% (7/9) | 0.80 | 92.0% | −0.9 pp | 3.8% (21/13) | 0.23 |
| Belebele en (900) | 95.2% | 94.8% | −0.4 pp | 1.8% (10/6) | 0.45 | 93.6% | −1.7 pp | 3.0% (21/6) | 0.006 |
| Global-MMLU-Lite de (400) | 72.5% | 71.2% | −1.2 pp | 3.8% (10/5) | 0.30 | 68.5% | −4.0 pp | 9.0% (26/10) | 0.011 |
| Global-MMLU-Lite en (400) | 73.8% | 72.0% | −1.8 pp | 4.2% (12/5) | 0.14 | 72.0% | −1.8 pp | 5.8% (15/8) | 0.21 |

Flips ("Accuracy is Not All You Need", Dutta et al. 2024): questions that turn
from right to wrong or from wrong to right against FP8, in brackets the two
directions (right → wrong / wrong → right). On Global-MMLU-Lite en both builds
lose the same 1.8 points, but uniform 3-bit gets there with more flips.

95% Wilson intervals are in the JSON (about ±1.5 pp for Belebele, ±4.5 pp for
Global-MMLU-Lite). The shipped build gives the same answer as FP8 on 95-98% of
the questions. None of its differences is significant at these sizes; a loss of
1-2 points on Global-MMLU-Lite cannot be ruled out either. Uniform 3-bit loses
significantly on two of the four sets: the 6-bit attention, shared expert,
embedding and head are worth their 0.1 bits per weight.

These are likelihood scores without reasoning. They measure what the
quantization changes, not what Kolibri scores with thinking enabled; compare
them with each other, not with the original model card.

## Time to first token

`measure-kolibri-ttft.py`, server `start-mlx_kolibri.sh` (`PREFILL_STEP` 2048,
APC on, 2 entries), `reasoning_effort: low`, median of 2 runs.

| context | cold TTFT | prefill rate | warm TTFT (exact-APC follow-up) |
|---|---|---|---|
| 1k | 0.8 s | 1,620 t/s | 0.35 s |
| 8k | 5.3 s | 1,590 t/s | 0.4 s* |
| 32k | 24 s | 1,380 t/s | 0.6 s* |
| 64k | 59 s | 1,090 t/s | 1.4 s† |
| 96k | **109 s** | 885 t/s | no hit‡ |

Peak memory at 96k: 38.0 GiB.

\* First run only. † With `APC_MEMORY_RESERVE_GB=1.5` (now the default of
`start-mlx_kolibri.sh`), one run; cold 4.7 / 21 / 56 s and warm 0.37 / 0.57 /
1.4 s at 8k / 32k / 64k. ‡ Automatic reserve; not re-measured with 1.5.
Two limits of the prefix cache, both measured:

- **Follow-up turns from ~32k tokens up miss the cache when free RAM is
  short.** mlx-vlm keeps a snapshot in memory only while its headroom (the
  smaller of working set minus active memory and free + inactive RAM per
  `vm_stat` plus the mlx cache) covers a reserve, the prefill reserve and the
  snapshot. The automatic reserve is a tenth of the working set, 4 GiB; next
  to 33 GB of weights only 4-5 GiB of RAM are free, so 8k usually fits and
  32k usually does not (`memory_skips` rises, disk restores never happen).
  With a 1.5 GB reserve 8k, 32k and 64k hit; the working-set term still guards
  the Metal limit. The prefill reserve is recomputed per prompt: a 64k+
  prefill does not disable APC for later, shorter prompts (an earlier version
  of this page said so; `prefill_reserve_bytes` in `/metrics` only shows the
  last request).
- **With `reasoning_effort: none`, follow-up turns miss the final snapshot.**
  The template ends the generation prompt with an empty `<think></think>`
  block but renders earlier assistant turns without it; the sliding-window
  caches cannot rewind to the point of divergence. At best the 2048-token
  interval checkpoint hits (8k: 6,144 tokens restored, 1.7 s). `low`, `medium`
  and `high` keep the prefix stable.

## Reproduce

```sh
# TTFT: server running, fresh start, long contexts last
MEM_PROBE_INTERVAL=0.5 caffeinate -dimsu ./start-mlx_kolibri.sh
./measure-kolibri-ttft.py

# Quality: server stopped; FP8 release in ~/src/mlx/models/Kolibri-1-FP8
./measure-kolibri-quality.py prepare
./measure-kolibri-quality.py check --ckpt 3bit
./convert-kolibri.py ~/src/mlx/models/Kolibri-1-FP8 ~/src/mlx/models/Kolibri-1-MLX-3bit-uniform --other-bits 3
for c in fp8 3bit 3bit-uniform; do ./measure-kolibri-quality.py forward --ckpt $c; done
./measure-kolibri-quality.py report
```

Run time on the M5 Pro: FP8 reference 24 min for all sets (556k tokens, peak
7.4 GiB), each quantized arm ~17 min, report ~2 min.
