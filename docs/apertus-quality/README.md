# Apertus-v1.5-8B-MLX-8bit: quality loss and time to first token

Measurements 2026-10-08/09, Apple M5 Pro, 48 GB, `iogpu.wired_limit_mb=40960`,
mlx 0.32.3, mlx-vlm `apertus1p5` branch at `ec86ea21`. Method, sets and
statistics: [`docs/quality-method.md`](../quality-method.md).

| file | content |
|---|---|
| `quality-2026-10-09.json` | per set, arm and group: KL distribution and perplexity; multiple choice with paired intervals, equivalence and flips; data sources and hashes; versions |
| `per-question-2026-10-09.jsonl` | per question and arm: set, id, predicted letter, answer, correct |
| `quality-baselines-2026-10-10.json`, `per-question-baselines-2026-10-10.jsonl` | the same with the community builds and GGUFs of the baseline comparison (2026-10-10) |

## Arms

- **bf16**: the release (`swiss-ai/Apertus-v1.5-8B`), the reference
- **noise floor**: bf16 again, prefilled in 512-token pieces against a KV
  cache instead of in one pass; same weights, different order of operations
- **8-bit (shipped)**: the omni build, affine, group size 64, 8.5 bits per weight
- **4-bit RTN**: round-to-nearest control, group size 64, 4.5 bits per weight

Only the language model is measured; the image and audio tokenizers are
float32 in every arm.

## Distribution against bf16

KL divergence per token (mean / p99.9) and how often the most likely next
token is the same, on about 380k tokens of five kinds of text:

| set | noise floor | 8-bit (shipped) | 4-bit RTN |
|---|---|---|---|
| Wikipedia de/en | 0.0007 / 0.024 / 98.7% | 0.0024 / 0.072 / 97.7% | 0.2016 / 4.188 / 79.9% |
| Calibration v5 | 0.0008 / 0.033 / 98.7% | 0.0026 / 0.097 / 97.6% | 0.1919 / 4.983 / 80.7% |
| chat (oasst2) | 0.0007 / 0.035 / 98.7% | 0.0023 / 0.138 / 97.6% | 0.1968 / 6.630 / 80.0% |
| tool calling | 0.0003 / 0.050 / 99.7% | 0.0022 / 0.181 / 98.7% | 0.2068 / 10.160 / 89.3% |
| FLORES, 23 EU languages | 0.0008 / 0.029 / 98.6% | 0.0025 / 0.086 / 97.6% | 0.1944 / 4.543 / 80.1% |
| chat (oasst2), assistant turns | 0.0005 / 0.019 / 98.8% | 0.0016 / 0.058 / 97.8% | 0.1605 / 4.248 / 81.0% |
| tool calling, assistant turns | 0.0001 / 0.009 / 99.9% | 0.0005 / 0.037 / 99.5% | 0.0536 / 2.417 / 94.9% |

The 8-bit build stays at 3 to 7 times the noise floor on every set, also on
chat and tool calling, where Unsloth expects quants calibrated on Wikipedia to
fall behind. Its worst token in a thousand stays below 0.2 nats. The 4-bit
control's mean KL is about 80 times the 8-bit build's, with its worst tail on
tool calling.

8-bit, full distribution:

| set | tokens | mean | median | p90 | p99 | p99.9 | max | same top | PPL ref → arm |
|---|---|---|---|---|---|---|---|---|---|
| Wikipedia de/en | 82,055 | 0.0024 | 0.0009 | 0.005 | 0.025 | 0.072 | 2.74 | 97.7% | 7.05 → 7.05 |
| Calibration v5 | 113,963 | 0.0026 | 0.0008 | 0.006 | 0.028 | 0.097 | 1.61 | 97.6% | 6.75 → 6.75 |
| chat (oasst2) | 64,852 | 0.0023 | 0.0006 | 0.004 | 0.026 | 0.138 | 0.77 | 97.6% | 5.55 → 5.54 |
| tool calling | 48,854 | 0.0022 | 0.0000 | 0.003 | 0.043 | 0.181 | 0.51 | 98.7% | 2.88 → 2.85 |
| FLORES, 23 EU languages | 70,104 | 0.0025 | 0.0010 | 0.005 | 0.025 | 0.086 | 4.63 | 97.6% | 8.20 → 8.22 |

Perplexities are per Apertus token and not comparable with Kolibri's.

### Per EU language

FLORES passages, mean KL / same top. Irish is not in Belebele.

| language | noise floor | 8-bit (shipped) | 4-bit RTN |
|---|---|---|---|
| bul_Cyrl | 0.0007 / 98.8% | 0.0022 / 97.6% | 0.1789 / 80.8% |
| ces_Latn | 0.0008 / 98.5% | 0.0024 / 98.1% | 0.1908 / 80.6% |
| dan_Latn | 0.0008 / 98.9% | 0.0027 / 98.0% | 0.1937 / 80.1% |
| deu_Latn | 0.0007 / 98.7% | 0.0020 / 97.1% | 0.1812 / 80.3% |
| ell_Grek | 0.0007 / 98.7% | 0.0020 / 97.7% | 0.1573 / 82.3% |
| eng_Latn | 0.0007 / 98.2% | 0.0022 / 97.4% | 0.2570 / 76.3% |
| est_Latn | 0.0012 / 98.0% | 0.0038 / 96.4% | 0.2779 / 75.7% |
| fin_Latn | 0.0008 / 98.9% | 0.0028 / 97.8% | 0.2200 / 79.4% |
| fra_Latn | 0.0006 / 98.5% | 0.0019 / 97.7% | 0.1610 / 78.2% |
| hrv_Latn | 0.0010 / 98.5% | 0.0029 / 97.8% | 0.2236 / 78.5% |
| hun_Latn | 0.0007 / 98.6% | 0.0024 / 97.6% | 0.1666 / 81.9% |
| ita_Latn | 0.0009 / 98.9% | 0.0020 / 97.7% | 0.1757 / 79.0% |
| lit_Latn | 0.0009 / 98.8% | 0.0036 / 97.5% | 0.2484 / 79.3% |
| lvs_Latn | 0.0010 / 98.7% | 0.0029 / 97.7% | 0.2310 / 80.7% |
| mlt_Latn | 0.0007 / 98.8% | 0.0029 / 97.9% | 0.2018 / 81.9% |
| nld_Latn | 0.0007 / 98.7% | 0.0020 / 97.9% | 0.1738 / 81.6% |
| pol_Latn | 0.0006 / 98.7% | 0.0022 / 97.8% | 0.1554 / 81.7% |
| por_Latn | 0.0007 / 98.8% | 0.0019 / 97.9% | 0.1552 / 79.2% |
| ron_Latn | 0.0006 / 98.7% | 0.0021 / 98.1% | 0.1802 / 80.3% |
| slk_Latn | 0.0008 / 98.5% | 0.0024 / 97.4% | 0.1792 / 81.8% |
| slv_Latn | 0.0009 / 98.5% | 0.0028 / 97.5% | 0.2163 / 79.2% |
| spa_Latn | 0.0010 / 98.1% | 0.0039 / 97.2% | 0.1634 / 78.5% |
| swe_Latn | 0.0009 / 98.6% | 0.0024 / 97.6% | 0.1872 / 79.6% |

The 8-bit build is close to uniform across languages (mean KL 0.0019 to
0.0039; highest for Spanish, Estonian and Lithuanian, where the noise floor is
also higher). Under 4-bit, Estonian,
English and Lithuanian move most.

## Multiple choice

Zero-shot through the chat template, scored by the letter logits. Δ in
percentage points against bf16 with a paired 95% interval (Newcombe);
"equiv." is a TOST with a margin of ±1 point fixed in advance; flips are
right → wrong / wrong → right. The noise-floor arm gives the same answer as
bf16 on every question (0 flips on all sets).

| set | bf16 | 8-bit (shipped) | Δ (95% CI) | equiv. | flips −/+ | 4-bit RTN | Δ (95% CI) | equiv. | flips −/+ |
|---|---|---|---|---|---|---|---|---|---|
| Belebele de (900) | 82.6% | 82.9% | +0.3 (−0.5 to +1.1) | yes | 4/7 | 73.7% | −8.9 (−11.2 to −6.7) | no | 96/16 |
| Belebele en (900) | 89.3% | 89.3% | +0.0 (−0.7 to +0.7) | yes | 3/3 | 85.2% | −4.1 (−6.0 to −2.3) | no | 53/16 |
| Global-MMLU-Lite de (400) | 62.0% | 61.8% | −0.2 (−1.5 to +1.0) | no | 3/2 | 56.5% | −5.5 (−9.1 to −1.8) | no | 39/17 |
| Global-MMLU-Lite en (400) | 68.0% | 68.0% | +0.0 (−1.1 to +1.1) | yes | 2/2 | 63.5% | −4.5 (−8.2 to −0.8) | no | 37/19 |
| … de, culturally sensitive (200) | 59.5% | 58.5% | −1.0 (−3.2 to +1.2) | no | 3/1 | 52.0% | −7.5 (−13.2 to −1.7) | no | 25/10 |
| … de, culturally agnostic (200) | 64.5% | 65.0% | +0.5 (−1.0 to +2.0) | no | 0/1 | 61.0% | −3.5 (−8.0 to +1.1) | no | 14/7 |
| … en, culturally sensitive (200) | 67.0% | 68.0% | +1.0 (−0.8 to +2.8) | no | 0/2 | 62.5% | −4.5 (−10.0 to +1.0) | no | 20/11 |
| … en, culturally agnostic (200) | 69.0% | 68.0% | −1.0 (−2.8 to +0.8) | no | 2/0 | 64.5% | −4.5 (−9.4 to +0.5) | no | 17/8 |
| Belebele Maltese (900) | 72.0% | 72.6% | +0.6 (−0.4 to +1.5) | no | 6/11 | 60.1% | −11.9 (−14.4 to −9.4) | no | 125/18 |
| Belebele Latvian (900) | 81.2% | 81.1% | −0.1 (−0.8 to +0.6) | yes | 5/4 | 71.7% | −9.6 (−12.1 to −7.1) | no | 113/27 |
| Belebele Estonian (900) | 79.0% | 78.8% | −0.2 (−1.2 to +0.7) | yes | 9/7 | 67.6% | −11.4 (−14.0 to −8.8) | no | 128/25 |
| Belebele Lithuanian (900) | 81.2% | 81.0% | −0.2 (−1.3 to +0.8) | no | 11/9 | 72.6% | −8.7 (−11.3 to −6.1) | no | 112/34 |
| Belebele de, options shifted (900) | 81.8% | 81.7% | −0.1 (−0.8 to +0.6) | yes | 4/3 | 72.0% | −9.8 (−12.0 to −7.5) | no | 101/13 |
| Belebele en, options shifted (900) | 88.8% | 88.8% | +0.0 (−0.9 to +0.9) | yes | 6/6 | 84.1% | −4.7 (−6.5 to −2.9) | no | 56/14 |
| Global-MMLU-Lite de, options shifted (400) | 61.5% | 62.0% | +0.5 (−0.8 to +1.8) | no | 2/4 | 54.2% | −7.2 (−11.2 to −3.2) | no | 49/20 |
| Global-MMLU-Lite en, options shifted (400) | 67.5% | 67.5% | +0.0 (−0.9 to +0.9) | yes | 1/1 | 64.2% | −3.2 (−6.6 to +0.1) | no | 30/17 |

For the 8-bit build, equivalence within ±1 point holds on Belebele de and en,
Global-MMLU-Lite en, Latvian and Estonian, and on the shifted versions where
the sample allows it. On Global-MMLU-Lite de, Maltese and Lithuanian no
difference is detectable, but the intervals reach just past ±1 point. Its
flips are balanced everywhere. The 4-bit control loses 4 to 12 points with
flips running clearly one way; it loses most in the less-represented
languages (Maltese −11.9, Estonian −11.4, Latvian −9.6, Lithuanian −8.7
points), more than in German (−8.9) and English (−4.1). On
Global-MMLU-Lite de it loses most on the culturally sensitive questions
(−7.5 points).

The four less-represented languages use the English instruction with
passage, question and options in the language (Irish is not in Belebele).

### Position bias

The four German and English sets were run again with the options shifted by
one position (A→B … D→A, answers remapped). "Same option chosen" is the share
of questions where the model picks the same content in both orders.

| set | arm | accuracy | options shifted | same option chosen | predicted A/B/C/D |
|---|---|---|---|---|---|
| Belebele de (900) | bf16 | 82.6% | 81.8% | 90.3% | 24% / 28% / 27% / 21% |
| Belebele de (900) | 8-bit (shipped) | 82.9% | 81.7% | 90.7% | 25% / 27% / 26% / 22% |
| Belebele de (900) | 4-bit RTN | 73.7% | 72.0% | 82.2% | 24% / 27% / 26% / 23% |
| Belebele de (900) | correct answers | | | | 23% / 28% / 27% / 22% |
| Belebele en (900) | bf16 | 89.3% | 88.8% | 94.0% | 24% / 27% / 26% / 23% |
| Belebele en (900) | 8-bit (shipped) | 89.3% | 88.8% | 93.2% | 24% / 27% / 26% / 22% |
| Belebele en (900) | 4-bit RTN | 85.2% | 84.1% | 88.6% | 24% / 26% / 28% / 22% |
| Belebele en (900) | correct answers | | | | 23% / 28% / 27% / 22% |
| Global-MMLU-Lite de (400) | bf16 | 62.0% | 61.5% | 79.0% | 22% / 33% / 23% / 22% |
| Global-MMLU-Lite de (400) | 8-bit (shipped) | 61.8% | 62.0% | 80.5% | 22% / 33% / 23% / 22% |
| Global-MMLU-Lite de (400) | 4-bit RTN | 56.5% | 54.2% | 68.8% | 20% / 34% / 26% / 21% |
| Global-MMLU-Lite de (400) | correct answers | | | | 24% / 27% / 24% / 25% |
| Global-MMLU-Lite en (400) | bf16 | 68.0% | 67.5% | 79.2% | 25% / 30% / 25% / 20% |
| Global-MMLU-Lite en (400) | 8-bit (shipped) | 68.0% | 67.5% | 79.5% | 26% / 29% / 25% / 20% |
| Global-MMLU-Lite en (400) | 4-bit RTN | 63.5% | 64.2% | 76.0% | 22% / 31% / 26% / 22% |
| Global-MMLU-Lite en (400) | correct answers | | | | 24% / 27% / 24% / 25% |

Even bf16 picks a different option after the shift on 6 to 21% of the
questions, so part of every multiple-choice score here depends on the order;
the 8-bit build is as order-dependent as bf16, 4-bit RTN more so (69% instead
of 79% on Global-MMLU-Lite de). On Global-MMLU-Lite the model prefers "B"
(about 30–33% of answers against 27% correct "B"). Accuracy changes by at most
about one point between the orders.

These are likelihood scores without reasoning: they show what the
quantization changes and are not comparable with published benchmark scores.

## MMLU 5-shot

All 14,042 MMLU test questions in the original Hendrycks format, five solved
examples of the same subject in front, no chat template, scored by the letter
logits after "Answer:" (the larger of " A" and "A" per letter). The prompts are
`unsloth/studio_mmlu`; the 5-shot prefix runs once per subject and every
question continues from a copy of its cache (`measure-mmlu.py`; identical
answers to the full prompts on a 114-question check). **Harness check:**
Llama 3.1 8B Instruct in bf16 scores 68.3% here, against 68.2% that Unsloth
gives for a correct implementation.

| arm | MMLU 5-shot | Δ vs bf16 (95% CI) | equivalent ±1 pp | flips −/+ | (MMLU − 25) / GB |
|---|---|---|---|---|---|
| bf16 (18.4 GB) | 66.9% | | | | 2.28 |
| **8-bit, shipped (10.1 GB)** | 66.9% | −0.1 (−0.3 to +0.1) | **yes** | 125/115 | 4.16 |
| 4-bit RTN (5.8 GB) | 62.8% | −4.2 (−4.8 to −3.5) | no | 1,356/773 | 6.56 |

With 14,042 questions the paired interval is tight enough to show
equivalence: the 8-bit build scores like bf16, also per category (STEM,
humanities, social sciences, other: all within 0.4 points). The 4-bit control
gains the most per gigabyte but loses 4.2 points, most in social sciences (−5.1) and STEM (−4.7).
Sizes are the safetensors files on disk (the omni builds include the float32
image and audio tokenizers). Summary: `mmlu5-2026-10-10.json`.

## Divergence @32

Does the build follow bf16 over several tokens, not just one? 300 prompts, 60
each from Terminal-Bench 2.1 tasks, SWE-bench Verified issues, AIME and HMMT
2025 problems, Belebele passages in Chinese, Arabic, Hindi, Russian and
Japanese, and 1,200-word Wikipedia windows; 32 tokens decoded greedily through
the chat template (`measure-divergence.py`, prompt set hashed in the result).
Unsloth's Divergence-300 prompts are not public, so the numbers follow its
method but not its prompts.

| prompts | 8-bit: all 32 tokens identical | mean first divergence | 4-bit RTN: identical | mean first divergence |
|---|---|---|---|---|
| all (300) | **64.3%** | 25.0 | 6.7% | 6.5 |
| math | 88.3% | 30.4 | 25.0% | 16.7 |
| terminal tasks | 68.3% | 26.3 | 3.3% | 4.1 |
| non-Latin scripts | 60.0% | 23.4 | 5.0% | 7.9 |
| SWE issues | 53.3% | 22.4 | 0.0% | 1.4 |
| long documents | 51.7% | 22.5 | 0.0% | 2.1 |

With the 8-bit build two thirds of the continuations are token for token the
ones of bf16, and where they part, it is on average after about 23 tokens. The
4-bit control leaves the bf16 path almost at once on code issues and long
documents. There is no noise floor for this test yet: greedy decoding can also
part ways through rounding alone, so the 8-bit figure is a lower bound for how
closely it follows. Summary: `divergence-2026-10-10.json`.

## Baselines: what people actually use

Community builds from the Hub, measured 2026-10-10 against the same bf16
reference on the same sets (sovereign-models#18). MLX builds go through the
layer-streamed forward; GGUFs run in llama.cpp, which hands over its full
logits (`gguf-logits.cpp`, `measure-quality.py gguf`). The community builds
are text-only (131,072 tokens); KL is taken over the common text vocabulary.
Sizes are the weight files on disk; ours include the float32 image and audio
tokenizers, the others do not.

Mean KL against bf16 per set, same top token on Calibration v5, and the
multiple-choice difference in points (≡: equivalent within ±1 point, TOST):

| build | GB | Wikipedia | Calibration v5 | chat | tools | FLORES EU | same top | Belebele de / en | Global-MMLU-Lite de / en |
|---|---|---|---|---|---|---|---|---|---|
| m1rkocasu mxfp4 | 4.3 | 0.091 | 0.052 | 0.071 | 0.076 | 0.079 | 89.6% | −2.3 / −1.0 | −1.0 / +0.8 |
| m1rkocasu 4-bit DWQ | 4.5 | 0.139 | 0.136 | 0.134 | 0.151 | 0.134 | 83.7% | −7.1 / −3.4 | −4.2 / −3.0 |
| tokimoa 4-bit | 4.5 | 0.202 | 0.192 | 0.197 | 0.207 | 0.194 | 80.7% | −8.9 / −4.1 | −5.5 / −4.5 |
| GGUF Q4_K_M (Colby) | 5.1 | 0.089 | 0.089 | 0.087 | 0.089 | 0.083 | 86.9% | −1.1 / −1.2 | +1.0 / +0.8 |
| m1rkocasu 5-bit | 5.5 | 0.045 | 0.044 | 0.042 | 0.049 | 0.043 | 90.3% | +0.7 / −0.3 | +1.5 / +1.5 |
| 4-bit RTN (our control) | 5.8 | 0.202 | 0.192 | 0.197 | 0.207 | 0.194 | 80.7% | −8.9 / −4.1 | −5.5 / −4.5 |
| m1rkocasu 6-bit | 6.5 | 0.012 | 0.013 | 0.011 | 0.010 | 0.012 | 94.7% | +0.1 ≡ / −0.2 ≡ | −1.0 / +1.0 |
| GGUF Q8_0 (andreasmartin) | 8.6 | 0.0014 | 0.0016 | 0.0013 | 0.0012 | 0.0016 | 98.1% | −0.1 ≡ / +0.1 ≡ | +0.5 / +0.5 |
| **8-bit (shipped)** | 10.1 | 0.0024 | 0.0026 | 0.0023 | 0.0022 | 0.0025 | 97.6% | +0.3 ≡ / 0.0 ≡ | −0.2 / 0.0 ≡ |
| noise floor | – | 0.0007 | 0.0008 | 0.0007 | 0.0003 | 0.0008 | 98.7% | | |

References: Belebele de 82.6% / en 89.3%, Global-MMLU-Lite de 62.0% / en 68.0% (bf16).

- tokimoa's 4-bit gives the same numbers as our 4-bit RTN control: both are
  mlx's default round-to-nearest with group size 64. The control is what
  people get from a plain `mlx_lm.convert -q`.
- At 4–5 GB, mxfp4 and Q4_K_M halve the KL of RTN and lose 1–2 points in
  multiple choice instead of 4–9. DWQ (distilled scales) improves on RTN by a
  third but stays behind mxfp4, which is also smaller.
- At 8 bit, llama.cpp's Q8_0 is about 40% closer to bf16 than our affine
  8-bit; it scales blocks of 32 weights instead of groups of 64. Both are
  within a few times the noise floor and equivalent in multiple choice.

## Time to first token

Pending. The first run (2026-10-08/09) was disturbed: the machine fell to 1%
battery, hibernated, and measured while charging from 3%; cold 8k varied
between 7.5 and 21 s. It will be repeated on a charged machine. Two
findings of that run that do not depend on the clock:

- Follow-up turns hit the prefix cache up to 64k; with thinking on (tested at
  8k and 32k) as well.
- At 96k the follow-up misses, also with `APC_MEMORY_RESERVE_GB=1.5`. A 96k
  f16 KV cache is ~12 GiB on top of 34 GiB in use; the snapshot does not fit.
  `KV_BITS=8` halves it and is the next test.

## Reproduce

```sh
# Quality: no server running, one run at a time
PY=~/src/mlx/.venv-apertus/bin/python
$PY -m mlx_vlm convert --hf-path ~/src/mlx/models/Apertus-v1.5-8B \
    --mlx-path ~/src/mlx/models/Apertus-v1.5-8B-MLX-4bit -q --q-bits 4 --q-group-size 64 --dtype bfloat16
$PY ./measure-apertus-quality.py prepare
$PY ./measure-apertus-quality.py check --ckpt 8bit
for c in bf16 8bit 4bit; do $PY ./measure-apertus-quality.py forward --ckpt $c; done
$PY ./measure-apertus-quality.py forward --ckpt bf16 --arm bf16-chunked --chunk 512
$PY ./measure-apertus-quality.py report
./quality-tables.py ~/src/mlx/apertus-quality/results/quality-<date>.json --ref bf16 \
    --arms bf16-chunked,8bit,4bit --names "noise floor,8-bit (shipped),4-bit RTN" --mc-arms 8bit,4bit
```

Run time on the M5 Pro: forward ~25 min per arm for all sets (~1 M tokens),
the noise-floor arm ~35 min; report ~50 min (CPU).
