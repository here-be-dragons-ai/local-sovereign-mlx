# Apertus-v1.5-8B-MLX-8bit: quality loss and time to first token

Measurements 2026-10-08, Apple M5 Pro, 48 GB, `iogpu.wired_limit_mb=40960`,
mlx 0.32.3, mlx-vlm `apertus1p5` branch at `ec86ea21` (repo commit `de8702d`
plus `measure-apertus-quality.py`). Same instrument and data as
[`docs/kolibri-quality/`](../kolibri-quality/README.md).

| file | content |
|---|---|
| `quality-2026-10-08.json` | KL / perplexity per arm and group, multiple-choice accuracy and flips, text sources (report re-run 2026-10-09 with p90 / p99.9 / max and flips; earlier values unchanged) |
| `per-question-2026-10-08.jsonl` | per question and arm: set, id, predicted letter, answer, correct |

## Quality

**Method.** Three arms go through the same layer-streamed forward pass
(`measure-apertus-quality.py`): the bf16 release (`swiss-ai/Apertus-v1.5-8B`)
as the reference, the shipped 8-bit omni build (affine, group size 64), and a
round-to-nearest 4-bit control (`mlx_vlm convert -q --q-bits 4`, group size
64). In both quantized arms the embedding and the LM head carry the same bits
as the decoder; the image and audio tokenizers are float32 and not measured.
The streamed pass is bit-identical to the ordinary in-memory forward on the
8-bit checkpoint (`check`: max |Δh| = 0). Logits are computed in fp32 on the
CPU from the final hidden state and each arm's own LM head.

**KL divergence and perplexity.** 82,055 scored tokens of Wikipedia prose at
pinned revisions (`sources` in the JSON), German and English, 4096-token
windows. Perplexities are per Apertus token and not comparable with
Kolibri's.

| arm | bits/weight | mean KL | median | p90 | p99 | p99.9 | max | same top | PPL (bf16: 7.047) |
|---|---|---|---|---|---|---|---|---|---|
| 8-bit (shipped) | 8.5 | 0.0024 | 0.0009 | 0.005 | 0.025 | 0.072 | 2.74 | 97.7% | 7.048 |
| 4-bit RTN | 4.5 | 0.202 | 0.090 | 0.471 | 1.83 | 4.19 | 17.4 | 79.9% | 8.210 |

Columns as `llama-perplexity --kl-divergence` reports them (see
[`docs/kolibri-quality/`](../kolibri-quality/README.md)). Even the worst token
in a thousand stays below 0.1 nats in the 8-bit build.

By language, 8-bit: German mean KL 0.0022, English 0.0026. 4-bit: German
0.180, English 0.223. The "pre" / "post" split of the text set is Kolibri's
knowledge cutoff (2026-06-18); it is in the JSON but means nothing here.

**Multiple choice.** Apertus' chat template with its default
"Deliberation: disabled", options A-D in the user turn, scored by the letter
logits at the first answer position. Δ and McNemar (exact, two-sided) are
paired against bf16.

| set | bf16 | 8-bit (shipped) | Δ | flips | p | 4-bit RTN | Δ | flips | p |
|---|---|---|---|---|---|---|---|---|---|
| Belebele de (900) | 82.6% | 82.9% | +0.3 pp | 1.2% (4/7) | 0.55 | 73.7% | −8.9 pp | 12.4% (96/16) | <0.001 |
| Belebele en (900) | 89.3% | 89.3% | ±0 pp | 0.7% (3/3) | 1.00 | 85.2% | −4.1 pp | 7.7% (53/16) | <0.001 |
| Global-MMLU-Lite de (400) | 62.0% | 61.8% | −0.2 pp | 1.2% (3/2) | 1.00 | 56.5% | −5.5 pp | 14.0% (39/17) | 0.005 |
| Global-MMLU-Lite en (400) | 68.0% | 68.0% | ±0 pp | 1.0% (2/2) | 1.00 | 63.5% | −4.5 pp | 14.0% (37/19) | 0.022 |

Flips: questions that turn right → wrong / wrong → right against bf16.

95% Wilson intervals are in the JSON. The 8-bit build gives the same answer as
bf16 on 98-99% of the questions; it is lossless within what these sets can
resolve. Plain 4-bit loses significantly on all four sets, most on German
reading comprehension, and agrees with bf16 on only 80-91% of the answers.
Apertus is sensitive to coarse rounding (see the 3-bit note in the 70B
script); a 4-bit build would need 6-bit embedding and head or AWQ scales, and
a measurement of its own.

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
# Quality: Apertus server stopped
PY=~/src/mlx/.venv-apertus/bin/python
$PY -m mlx_vlm convert --hf-path ~/src/mlx/models/Apertus-v1.5-8B \
    --mlx-path ~/src/mlx/models/Apertus-v1.5-8B-MLX-4bit -q --q-bits 4 --q-group-size 64 --dtype bfloat16
$PY ./measure-apertus-quality.py prepare
$PY ./measure-apertus-quality.py check --ckpt 8bit
for c in bf16 8bit 4bit; do $PY ./measure-apertus-quality.py forward --ckpt $c; done
$PY ./measure-apertus-quality.py report
```

Run time on the M5 Pro: ~26 min per arm for all sets (672k tokens, peak
3.9 GiB), report ~2 min.
