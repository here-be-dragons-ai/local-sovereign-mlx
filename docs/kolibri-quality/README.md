# Kolibri-1-MLX-3bit: quality loss and time to first token

Measurements 2026-10-08/09, Apple M5 Pro, 48 GB,
`iogpu.wired_limit_mb=40960`, mlx 0.32.2, mlx-vlm 0.7.4. Method, sets and
statistics: [`docs/quality-method.md`](../quality-method.md).

| file | content |
|---|---|
| `quality-2026-10-09.json` | per set, arm and group: KL distribution and perplexity; multiple choice with paired intervals, equivalence and flips; data sources and hashes; versions |
| `per-question-2026-10-09.jsonl` | per question and arm: set, id, predicted letter, answer, correct |
| `ttft-2026-10-08.jsonl` | one line per TTFT request, with server flags and versions |
| `ttft-reserve-2026-10-08.jsonl` | the same with `APC_MEMORY_RESERVE_GB=1.5`, one run up to 64k |

## Arms

- **FP8**: the release (`Aleph-Alpha/Kolibri-1`, block-FP8, dequantized to
  bf16 layer by layer), the reference
- **noise floor**: FP8 again, prefilled in 512-token pieces against a KV cache
  instead of in one pass; same weights, different order of operations
- **3/6-bit (shipped)**: 3.61 bits per weight
- **uniform 3-bit**: control, `convert-kolibri.py --other-bits 3`, 3.51 bits
  per weight

## The noise floor is high for this model

Run against itself through a different number of tokens per call, the FP8
release already differs noticeably from itself: mean KL 0.036 and p99.9 5.7
on Wikipedia, 95% same top token. For comparison, the same test on dense
Apertus 1.5 8B gives 0.0007 and 0.024. It is rounding, not a different
computation: after the first decoder layer the hidden states differ by about
1e-4 (bf16 rounding of a matmul over 512 instead of 2,000 rows), and a single
chunk of 2,048 tokens through the cache path is bit-identical to the
one-pass forward. Over 50 layers of top-6-of-384 expert routing these
differences tip expert choices and grow. The 3-bit build shows the same
effect on its own (KL 0.039 between the two paths). In practice two servers
with different prefill step sizes already produce somewhat different
Kolibri outputs. No multiple-choice answer changes from this noise (0 flips
on all sets).

## Distribution against FP8

KL divergence per token (mean / p99.9) and how often the most likely next
token is the same:

| set | noise floor | 3/6-bit (shipped) | uniform 3-bit |
|---|---|---|---|
| Wikipedia de/en | 0.0363 / 5.695 / 95.0% | 0.1140 / 10.344 / 88.9% | 0.3764 / 13.066 / 76.5% |
| Calibration v5 | 0.0938 / 8.570 / 91.8% | 0.2081 / 11.012 / 85.3% | 0.5253 / 12.811 / 73.1% |
| chat (oasst2) | 0.0106 / 1.315 / 97.0% | 0.3983 / 9.714 / 72.6% | 0.5273 / 10.498 / 68.9% |
| tool calling | 0.0421 / 6.760 / 97.4% | 0.1120 / 10.112 / 94.0% | 0.3101 / 13.479 / 88.0% |
| FLORES, 23 EU languages | 0.0605 / 4.495 / 89.9% | 0.1671 / 6.448 / 81.0% | 0.5145 / 8.320 / 65.8% |
| chat (oasst2), assistant turns | 0.0090 / 1.172 / 97.2% | 0.3569 / 9.642 / 73.3% | 0.4727 / 10.228 / 69.8% |
| tool calling, assistant turns | 0.0007 / 0.044 / 99.6% | 0.0047 / 0.253 / 98.6% | 0.0285 / 1.596 / 96.8% |

Against that floor the shipped build sits at 2 to 3 times the noise on
Wikipedia, Calibration v5, tool calling and the EU languages. **On chat it
does not**: mean KL 0.40 against a floor of 0.011, 73% same top token
against 97%, perplexity 11.6 → 13.5; the assistant turns alone look the
same. The heavy tail on Wikipedia (p99.9 10.3) is about half noise (5.7).
Uniform 3-bit is worse on every set.

Noise floor and shipped build, full distribution:

| set | tokens | mean | median | p90 | p99 | p99.9 | max | same top | PPL ref → arm |
|---|---|---|---|---|---|---|---|---|---|
| Wikipedia de/en | 75,900 | 0.0363 | 0.0030 | 0.027 | 0.523 | 5.695 | 19.31 | 95.0% | 16.98 → 17.00 |
| Calibration v5 | 119,611 | 0.0938 | 0.0070 | 0.081 | 2.156 | 8.570 | 19.29 | 91.8% | 16.02 → 16.03 |
| chat (oasst2) | 64,821 | 0.0106 | 0.0008 | 0.009 | 0.110 | 1.315 | 15.45 | 97.0% | 11.61 → 11.61 |
| tool calling | 59,259 | 0.0421 | 0.0000 | 0.011 | 0.831 | 6.760 | 19.02 | 97.4% | 4.46 → 4.45 |
| FLORES, 23 EU languages | 90,846 | 0.0605 | 0.0156 | 0.092 | 0.853 | 4.495 | 14.21 | 89.9% | 24.68 → 24.82 |

| set | tokens | mean | median | p90 | p99 | p99.9 | max | same top | PPL ref → arm |
|---|---|---|---|---|---|---|---|---|---|
| Wikipedia de/en | 75,900 | 0.1140 | 0.0211 | 0.133 | 2.074 | 10.344 | 19.78 | 88.9% | 16.98 → 17.11 |
| Calibration v5 | 119,611 | 0.2081 | 0.0320 | 0.286 | 4.322 | 11.012 | 19.81 | 85.3% | 16.02 → 16.39 |
| chat (oasst2) | 64,821 | 0.3983 | 0.1659 | 0.922 | 4.029 | 9.714 | 14.92 | 72.6% | 11.61 → 13.45 |
| tool calling | 59,259 | 0.1120 | 0.0013 | 0.085 | 2.766 | 10.112 | 24.08 | 94.0% | 4.46 → 4.36 |
| FLORES, 23 EU languages | 90,846 | 0.1671 | 0.0701 | 0.327 | 1.874 | 6.448 | 15.29 | 81.0% | 24.68 → 26.35 |

Cross-runtime: `llama-perplexity -c 4096` on the same text gives **17.75 ±
0.30** for the Q3_K_S GGUF (`Eliasfpv28/Kolibri-1-Q3_K_S-GGUF`, 31.5 GiB). In the same convention -- second half of each window scored -- the
FP8 reference is 15.69, the shipped 3/6-bit build 15.75, uniform 3-bit 17.63.
The windows are not cut identically (llama.cpp splits across article
boundaries), so the numbers are comparable, not equal. No KL for the GGUF: that
needs llama.cpp's logit file format for the reference.

### Per EU language

FLORES passages, mean KL / same top. Irish is not in Belebele.

| language | noise floor | 3/6-bit (shipped) | uniform 3-bit |
|---|---|---|---|
| bul_Cyrl | 0.0806 / 88.8% | 0.2102 / 80.2% | 0.4795 / 66.9% |
| ces_Latn | 0.0507 / 91.3% | 0.1458 / 83.0% | 0.4789 / 67.3% |
| dan_Latn | 0.0714 / 87.4% | 0.2004 / 78.5% | 0.6503 / 60.4% |
| deu_Latn | 0.0460 / 92.6% | 0.1049 / 86.5% | 0.3810 / 71.5% |
| ell_Grek | 0.0581 / 89.8% | 0.1235 / 83.6% | 0.3267 / 72.5% |
| eng_Latn | 0.0725 / 91.5% | 0.1702 / 85.0% | 0.4737 / 71.5% |
| est_Latn | 0.0855 / 83.5% | 0.2476 / 70.8% | 0.7159 / 53.8% |
| fin_Latn | 0.0431 / 88.9% | 0.1605 / 77.9% | 0.5770 / 59.1% |
| fra_Latn | 0.1145 / 90.2% | 0.1880 / 83.9% | 0.5357 / 70.9% |
| hrv_Latn | 0.0439 / 89.9% | 0.1371 / 81.1% | 0.5027 / 62.8% |
| hun_Latn | 0.0409 / 91.2% | 0.1601 / 80.1% | 0.5644 / 62.5% |
| ita_Latn | 0.0717 / 90.4% | 0.1759 / 83.2% | 0.4963 / 71.0% |
| lit_Latn | 0.0370 / 90.0% | 0.1628 / 76.6% | 0.5134 / 61.0% |
| lvs_Latn | 0.0512 / 90.2% | 0.1628 / 78.3% | 0.5494 / 60.6% |
| mlt_Latn | 0.0337 / 90.3% | 0.1803 / 78.3% | 0.6158 / 60.7% |
| nld_Latn | 0.0630 / 93.2% | 0.1525 / 86.7% | 0.5223 / 71.1% |
| pol_Latn | 0.0582 / 90.9% | 0.1488 / 84.1% | 0.3948 / 72.3% |
| por_Latn | 0.0635 / 91.5% | 0.1390 / 85.9% | 0.4340 / 72.6% |
| ron_Latn | 0.0718 / 89.8% | 0.2140 / 79.8% | 0.6619 / 61.8% |
| slk_Latn | 0.0525 / 89.5% | 0.1632 / 79.5% | 0.4746 / 67.2% |
| slv_Latn | 0.0496 / 89.1% | 0.1580 / 79.7% | 0.5450 / 61.3% |
| spa_Latn | 0.0920 / 90.0% | 0.1874 / 83.6% | 0.5362 / 71.2% |
| swe_Latn | 0.0613 / 88.5% | 0.1706 / 79.7% | 0.5492 / 64.5% |

## Multiple choice

Zero-shot through the chat template with `reasoning_effort=none`, scored by
the letter logits. Δ in percentage points against FP8 with a paired 95%
interval (Newcombe); "equiv." is a TOST with a margin of ±1 point fixed in
advance; flips are right → wrong / wrong → right.

| set | fp8 | 3/6-bit (shipped) | Δ (95% CI) | equiv. | flips −/+ | uniform 3-bit | Δ (95% CI) | equiv. | flips −/+ |
|---|---|---|---|---|---|---|---|---|---|
| Belebele de (900) | 92.9% | 93.1% | +0.2 (−0.7 to +1.2) | no | 7/9 | 92.0% | −0.9 (−2.2 to +0.4) | no | 21/13 |
| Belebele en (900) | 95.2% | 94.8% | −0.4 (−1.4 to +0.5) | no | 10/6 | 93.6% | −1.7 (−2.9 to −0.5) | no | 21/6 |
| Global-MMLU-Lite de (400) | 72.5% | 71.2% | −1.2 (−3.2 to +0.7) | no | 10/5 | 68.5% | −4.0 (−7.0 to −1.0) | no | 26/10 |
| Global-MMLU-Lite en (400) | 73.8% | 72.0% | −1.8 (−3.9 to +0.3) | no | 12/5 | 72.0% | −1.8 (−4.2 to +0.7) | no | 15/8 |
| … de, culturally sensitive (200) | 67.5% | 64.5% | −3.0 (−6.3 to +0.2) | no | 8/2 | 63.0% | −4.5 (−8.8 to −0.2) | no | 14/5 |
| … de, culturally agnostic (200) | 77.5% | 78.0% | +0.5 (−2.1 to +3.1) | no | 2/3 | 74.0% | −3.5 (−7.7 to +0.7) | no | 12/5 |
| … en, culturally sensitive (200) | 68.5% | 66.5% | −2.0 (−5.6 to +1.5) | no | 8/4 | 65.5% | −3.0 (−6.8 to +0.8) | no | 10/4 |
| … en, culturally agnostic (200) | 79.0% | 77.5% | −1.5 (−4.1 to +1.0) | no | 4/1 | 78.5% | −0.5 (−3.8 to +2.7) | no | 5/4 |
| Belebele Maltese (900) | 46.1% | 48.1% | +2.0 (−0.1 to +4.1) | no | 37/55 | – | – | – | – |
| Belebele Latvian (900) | 65.3% | 66.1% | +0.8 (−1.1 to +2.6) | no | 32/39 | – | – | – | – |
| Belebele Estonian (900) | 58.9% | 59.6% | +0.7 (−1.2 to +2.5) | no | 32/38 | – | – | – | – |
| Belebele Lithuanian (900) | 63.0% | 63.6% | +0.6 (−1.1 to +2.2) | no | 25/30 | – | – | – | – |
| Belebele de, options shifted (900) | 92.3% | 92.1% | −0.2 (−1.0 to +0.6) | yes | 6/4 | – | – | – | – |
| Belebele en, options shifted (900) | 94.1% | 93.8% | −0.3 (−1.0 to +0.3) | yes | 4/1 | – | – | – | – |
| Global-MMLU-Lite de, options shifted (400) | 69.0% | 67.2% | −1.8 (−4.2 to +0.7) | no | 15/8 | – | – | – | – |
| Global-MMLU-Lite en, options shifted (400) | 73.5% | 71.8% | −1.8 (−4.4 to +0.9) | no | 17/10 | – | – | – | – |

Equivalence within ±1 point holds only for the shifted Belebele sets; elsewhere
the intervals are wider than the margin. For the shipped build no difference
is detectable on any set; its flips lean slightly towards losses on
Global-MMLU-Lite (10/5, 12/5), most on the culturally sensitive German
questions (8/2, −3.0 points, interval −6.3 to +0.2). Uniform 3-bit loses
significantly on Belebele en and Global-MMLU-Lite de. The uniform control was
not run on the sets added later (languages, shifted options).

**Less-represented languages.** Kolibri is trained for German and English and
it shows: Maltese 46%, Estonian 59%, Lithuanian 63%, Latvian 65% on Belebele,
against 93% in German. The shipped build is not worse there than FP8 (+0.6 to
+2.0 points, not significant); in Maltese its flips even run towards correct
answers (37 lost, 55 gained). The instruction is English, passage, question
and options are in the language (Irish is not in Belebele).

### Position bias

The four German and English sets were run again with the options shifted by
one position (A→B … D→A, answers remapped). "Same option chosen" is the share
of questions where the model picks the same content in both orders.

| set | arm | accuracy | options shifted | same option chosen | predicted A/B/C/D |
|---|---|---|---|---|---|
| Belebele de (900) | fp8 | 92.9% | 92.3% | 94.3% | 24% / 28% / 26% / 22% |
| Belebele de (900) | 3/6-bit (shipped) | 93.1% | 92.1% | 94.9% | 24% / 29% / 26% / 22% |
| Belebele de (900) | correct answers | | | | 23% / 28% / 27% / 22% |
| Belebele en (900) | fp8 | 95.2% | 94.1% | 96.3% | 24% / 28% / 27% / 21% |
| Belebele en (900) | 3/6-bit (shipped) | 94.8% | 93.8% | 95.9% | 23% / 28% / 27% / 22% |
| Belebele en (900) | correct answers | | | | 23% / 28% / 27% / 22% |
| Global-MMLU-Lite de (400) | fp8 | 72.5% | 69.0% | 78.8% | 21% / 35% / 25% / 19% |
| Global-MMLU-Lite de (400) | 3/6-bit (shipped) | 71.2% | 67.2% | 77.5% | 21% / 37% / 24% / 18% |
| Global-MMLU-Lite de (400) | correct answers | | | | 24% / 27% / 24% / 25% |
| Global-MMLU-Lite en (400) | fp8 | 73.8% | 73.5% | 81.2% | 20% / 34% / 25% / 21% |
| Global-MMLU-Lite en (400) | 3/6-bit (shipped) | 72.0% | 71.8% | 80.0% | 22% / 34% / 25% / 20% |
| Global-MMLU-Lite en (400) | correct answers | | | | 24% / 27% / 24% / 25% |

Kolibri is more stable than Apertus on Belebele (94–96% same option) and as
order-dependent on Global-MMLU-Lite (78–81%), where it prefers "B" (34–37% of
answers against 27% correct). The shipped build behaves like FP8. Accuracy on
Global-MMLU-Lite de drops by 3.5 points for FP8 and 4.0 for the shipped build
when the options are shifted, a sign of that preference.

These are likelihood scores without reasoning: they show what the
quantization changes and are not comparable with the scores on the original
model card.

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

# Quality: no server running, one run at a time; FP8 release in ~/src/mlx/models/Kolibri-1-FP8
./measure-kolibri-quality.py prepare
./measure-kolibri-quality.py check --ckpt 3bit
./convert-kolibri.py ~/src/mlx/models/Kolibri-1-FP8 ~/src/mlx/models/Kolibri-1-MLX-3bit-uniform --other-bits 3
for c in fp8 3bit 3bit-uniform; do ./measure-kolibri-quality.py forward --ckpt $c; done
./measure-kolibri-quality.py forward --ckpt fp8 --arm fp8-chunked --chunk 512
./measure-kolibri-quality.py report
./quality-tables.py ~/src/mlx/kolibri-quality/results/quality-<date>.json --ref fp8 \
    --arms fp8-chunked,3bit,3bit-uniform --names "noise floor,3/6-bit (shipped),uniform 3-bit" \
    --mc-arms 3bit,3bit-uniform
```

`forward` keeps each arm's LM head next to its hidden states, so `report`
no longer needs the checkpoints (the uniform 3-bit control can be deleted
after its forward pass). Run time on the M5 Pro: forward ~30 min per arm for
all sets, the noise-floor arm ~1.5 h; report ~1 h (CPU).
