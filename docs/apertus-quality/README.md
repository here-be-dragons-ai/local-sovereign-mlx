# Apertus-v1.5-8B-MLX-8bit: quality loss and time to first token

Measurements 2026-10-08/09, Apple M5 Pro, 48 GB, `iogpu.wired_limit_mb=40960`,
mlx 0.32.3, mlx-vlm `apertus1p5` branch at `ec86ea21`. Method, sets and
statistics: [`docs/quality-method.md`](../quality-method.md).

| file | content |
|---|---|
| `quality-2026-10-09.json` | per set, arm and group: KL distribution and perplexity; multiple choice with paired intervals, equivalence and flips; data sources and hashes; versions |
| `per-question-2026-10-09.jsonl` | per question and arm: set, id, predicted letter, answer, correct |

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

For the 8-bit build, equivalence within ±1 point holds on Belebele de and en
and Global-MMLU-Lite en. On Global-MMLU-Lite de no difference is detectable,
but 400 questions are too few to show equivalence within ±1 point; the same
holds for the 200-question subgroups. Its flips are balanced. The 4-bit
control loses 4 to 9 points with flips running clearly one way (e.g. 96 to 16
on Belebele de), a systematic loss rather than noise; on Global-MMLU-Lite de
it loses most on the culturally sensitive questions (−7.5 points).

These are likelihood scores without reasoning: they show what the
quantization changes and are not comparable with published benchmark scores.

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
