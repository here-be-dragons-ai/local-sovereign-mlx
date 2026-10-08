#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# mlx-vlm server start script  -  swiss-ai Apertus 1.5 70B, text only, 3 bit
# TARGET HARDWARE:  Apple Silicon, 48 GB, iogpu.wired_limit_mb=40960
#
# Thin wrapper around start-mlx_apertus.sh with the settings the 70B needs.
#
# Checkpoint: Apertus-v1.5-70B-MLX-3bit, from the gated bf16 release with
# ~/src/mlx/apertus-ref/run-70b.sh: decoder linears 3 bit, embed_tokens and
# lm_head 6 bit (affine, group 64), image / audio tokenizer dropped. Plain
# round-to-nearest 3 bit breaks Apertus (up_proj in front of xIELU), so the
# weights carry AWQ scales computed layer by layer (awq-apertus-scales.py).
#
# ── MEMORY ──────────────────────────────────────────────────────────────────
#   weights        30.4 GiB
#   KV cache       320 KiB/token in f16 (80 layers x 8 KV heads x 128 x 2 x 2 B),
#                  ~180 KiB/token at 8 bit -> KV_BITS=8 is the default here
#   APC            no exact-cache snapshots in RAM (each is another full KV
#                  copy, ~5.6 GiB at 32k); prefix blocks and the disk tier stay
#   Clients should stay at 32768 tokens of context.
#
# ── SPEED (measured 2026-10-08, M5 Pro) ─────────────────────────────────────
#   decode    ~8.7 t/s;  NLL 2.000 on the 8B eval text (8B bf16: 2.070)
#
# ── REASONING / TOOLS ───────────────────────────────────────────────────────
#   As for the 8B: top-level enable_thinking, <|inner_prefix|> markers,
#   tool calls parsed by the apertus parser.
# ─────────────────────────────────────────────────────────────────────────────

export MODEL_DIR="${MODEL_DIR:-${MLX_MODELS:-$HOME/src/mlx/models}/Apertus-v1.5-70B-MLX-3bit}"
export MODEL_ALIAS="${MODEL_ALIAS:-Apertus-v1.5-70B-local}"
export STATE_DIR="${STATE_DIR:-$HOME/.mlx-apertus70b}"
export PORT="${PORT:-8890}"
export KV_BITS="${KV_BITS:-8}"
export APC_ENTRIES="${APC_ENTRIES:-0}"
export PREFILL_STEP="${PREFILL_STEP:-1024}"

exec "${0:A:h}/start-mlx_apertus.sh" "$@"
