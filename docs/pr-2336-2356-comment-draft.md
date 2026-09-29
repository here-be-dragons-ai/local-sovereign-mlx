# Draft comments for upstream PRs #2336 and #2356

Status: **posted 2026-09-26** as gtonic, verbatim from the sections below:
- #2336: https://github.com/Blaizzy/mlx-vlm/pull/2336#issuecomment-5846397310
- #2356: https://github.com/Blaizzy/mlx-vlm/pull/2356#issuecomment-5846397461

Measured 2026-09-24 (0.7.2) and 2026-09-25 (0.7.3).

---

## For #2336

Independent measurement on a real checkpoint, since the PR deliberately claims
no throughput number: on this setup it is the fix for #2210.

**Setup:** Apple M5 Pro 48 GB, macOS 26, mlx 0.32.2, mlx-vlm 0.7.3 (also
measured on 0.7.2), `mlx-community/Qwen3.8-27B-4bit` (`qwen3_5`, 16
full-attention layers), server with continuous batching, **one** sequence,
exact APC with disk tier, f16 KV, **no drafter**. 26,690-token prompt, 300
decoded tokens, `temperature 0`, pairs of identical prompts where only the cold
arm carries a nonce, one server restart per arm.

| arm | cold tok/s | warm (APC hit) tok/s | warm/cold | warm `active+cache` |
|---|---:|---:|---:|---:|
| 0.7.3 | 16.08 / 16.08 | 10.49 / 10.57 | **0.655** | 28.8 / 26.8 GiB |
| 0.7.3 + this PR | 16.02 / 16.04 | 16.05 / 16.06 | **1.001** | 22.2 / 22.2 GiB |

0.7.2 gave the same picture (0.658 → 0.998). Greedy output is bit-identical with
and without the PR (4k prompt, 250 tokens, cold and warm).

Two observations that may help review:

1. **The gate matches more than "a batch that shrank to one".**
   `_is_single_row_batch_cache` matches any single-row, unquantized
   `BatchKVCache`, so on a single-slot server every request restored from APC
   takes the extract/merge path on every decode token — which is exactly
   #2210. Note that `extract()` returns an exactly-sized `KVCache`, so the
   following `update_and_fetch` must reallocate and concatenate as well: three
   passes over the prefix per layer per step, not two. In isolation (16 layers
   × 4 KV heads × 256, bf16, one token) that is 10.5 ms at 8k, 29.5 ms at 26.7k
   and 57.9 ms at 50k per forward, against under 1 ms borrowed.
2. **Speculative decoding never reaches this branch.** The shortcut requires
   `hidden_sink is None`, and DFlash/MTP verify passes always pass
   `capture_layer_ids`, so `hidden_sink` is a list. With DFlash 2 loaded the
   warm/cold ratio was 1.00 before this PR and stays there (cold 21.6–22.5 tok/s
   with the PR, unchanged). That explains why #2210 disappears as soon as a
   drafter is on, which I had reported there without knowing why.

---

## For #2356

Independent check on other hardware, as requested: it reproduces your result.

**Setup:** Apple M5 Pro 48 GB (not an M2 Ultra), macOS 26, mlx 0.32.2, mlx-vlm
0.7.3 + this PR (library change only, tests not installed),
`mlx-community/Qwen3.8-27B-4bit` unmodified, server, one sequence, exact APC
with disk tier, f16 KV, no drafter. 26,690-token restored prefix — about twice
your longest row — 300 decoded tokens, `temperature 0`, one restart per arm.

| arm | cold tok/s | restored tok/s | restored/cold | warm `active+cache` |
|---|---:|---:|---:|---:|
| 0.7.3 | 16.08 / 16.08 | 10.49 / 10.57 | **0.655** | 28.8 / 26.8 GiB |
| 0.7.3 + #2356 | 15.84 / 16.06 | 15.82 / 16.12 | **1.001** | 20.6 / 20.6 GiB |
| 0.7.3 + #2336 (for comparison) | 16.02 / 16.04 | 16.05 / 16.06 | **1.001** | 22.2 / 22.2 GiB |

Greedy output is bit-identical across all three arms. The memory column is the
part your table doesn't show: the restored arm on 0.7.3 sits 3–8 GiB above the
cold one at this length, and this PR removes that too.

On your question whether the batch caches on this path are intentional: #2336
fixes the same symptom inside the model (it borrows the arrays of a one-row
`BatchKVCache` instead of extract/merge), and it additionally covers a batch
that *shrinks* to one row, which `merge_rows` never sees. The two look
complementary rather than competing. With a speculative drafter neither
matters: the verify pass carries `capture_layer_ids`, which skips the
single-row shortcut entirely.

---

## Reply on #2356 to fblissjr (2026-09-28: "Do you have the cold arm's memory too?")

Status: **posted 2026-09-29** as gtonic, verbatim from the Reply section:
https://github.com/Blaizzy/mlx-vlm/pull/2356#issuecomment-5885658484

Measured 2026-09-29, raw output below the
reply. Same setup as before, same version as the posted table (0.7.3, mlx
0.32.2, the other seven local patches in place), one restart and a fresh
`STATE_DIR` per arm, three pairs per arm.

### Reply

Now I do — and it corrects my table. The `active+cache` column there was the
max over the whole request, sampled every 5 s, so a cold arm would have
reported its prefill, and the 20.6 vs 22.2 GiB between this PR and #2336 was
sampling noise. Re-measured at 0.5 s and split at the decode start:

| arm | cold prefill | cold decode | restored decode | restored/cold tok/s |
|---|---:|---:|---:|---:|
| 0.7.3 | 22.2–22.9 + 2.3 | 20.58 + 1.65 | 20.84 + **12.1–12.7** | 0.656 |
| 0.7.3 + #2356 | 22.2–22.9 + 2.3–4.6 | 20.58 + 1.65 | 20.58 + 1.65 | 1.003 |
| 0.7.3 + #2336 | 22.2–23.0 + 2.2–2.3 | 20.58–20.67 + 1.65 | 20.59–20.68 + 1.65 | 1.009 |

GiB, max `active + cache` per phase, 26,690-token prefix, 300 decoded tokens,
no drafter.

So with either PR the restored row decodes in **exactly** the cold arm's
memory; neither has an edge. On 0.7.3 the excess is almost entirely allocator
cache, not live arrays: active is only +0.26 GiB over cold, cache is +10.5 GiB —
the per-token extract/merge copies being freed into the buffer cache. Not a
leak, but it holds on to ~10 GiB of GPU memory for the whole decode at this
length, and it's above the cold arm's prefill peak too.

Greedy output is still bit-identical across all three arms, cold and restored.

Agreed on the overlap: #2336 does catch the restored row as well, and I'd
missed that `merge_rows` only runs at insert. For us the memory question comes
out a tie, so the choice between them is about coverage (shrink-to-one) vs.
the plain cache on any model and the skipped merge copy on restore.

### Raw

`./measure-apc-warm-decode.py --prompt-tokens 28000 --max-tokens 300 --repeat 3
--phases`, server with `ENABLE_SPEC_DECODE=0 MEM_PROBE_INTERVAL=0.5`.
Per arm, pairs 1–3, restored decode: 10.41 / 10.51 / 10.59 t/s (0.7.3),
16.07 / 16.09 / 16.07 (#2356), 16.06 / 16.06 / 16.05 (#2336); cold 15.87–16.07
everywhere. `/v1/cache/stats` after each arm: `exact_hits` 4, `exact_stores` 8,
`memory_skips` 0.
