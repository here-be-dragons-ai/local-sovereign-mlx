#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# mlx-vlm server start script  -  swiss-ai Apertus 1.5 8B (text, image, audio)
# TARGET HARDWARE:  Apple Silicon, 48 GB, iogpu.wired_limit_mb=40960
#
# Checkpoint: Apertus-v1.5-8B-MLX-8bit-omni, converted from the gated
# swiss-ai/Apertus-v1.5-8B release with the apertus1p5 branch of mlx-vlm:
# language model 8 bit affine (group size 64), vision tokenizer (EMU3.5/IBQ)
# and audio tokenizer (WavTokenizer) kept in float32. 9.4 GB on disk.
# The model lives only on that branch (here-be-dragons-ai/mlx-vlm#1), so this
# script runs its own venv with an editable install of the worktree; the
# shared venv of the Qwen3.8 / Kolibri servers stays untouched.
#
# ── MEMORY ──────────────────────────────────────────────────────────────────
#   weights        ~9.4 GiB
#   KV cache       ~128 KiB/token in f16 (32 layers x 8 KV heads x 128 x 2 x 2 B)
#   images         640x480 = 1200 tokens, max 1.96 MP = ~7600 tokens; the
#                  image tokenizer adds up to ~12 GB while it encodes
#   Measured 2026-10-07, M5 Pro: peak 15 GB with a photo, 22 GB with the
#   largest image, 14 GB with image + 8 s of audio.
#
# ── SPEED (measured 2026-10-07, M5 Pro, mlx 0.32.3) ─────────────────────────
#   decode    ~30-33 t/s
#   prefill   ~700-780 t/s
#   image     ~1 s (photo) to ~4 s (1.96 MP) to tokenize, on the GPU
#   audio     ~0.1 s per second of audio to tokenize, on the CPU
#
# ── INPUT ───────────────────────────────────────────────────────────────────
#   Images and audio go through the normal OpenAI chat format (image_url,
#   input_audio). Audio is read as 16-bit and resampled to 24 kHz.
#   The 8B model handles one modality per prompt well; with image AND audio
#   in one prompt it tends to ignore the audio (the reference does the same).
#
# ── REASONING ───────────────────────────────────────────────────────────────
#   Off by default ("Deliberation: disabled" in the template). Clients turn it
#   on per request with the top-level field `enable_thinking: true`. Apertus
#   marks the thinking with <|inner_prefix|> ... <|inner_suffix|>, not <think>.
#
# ── LICENCE ─────────────────────────────────────────────────────────────────
#   Apache-2.0 plus the Apertus AUP. Do not share the converted checkpoint
#   without the AUP conditions (express assent, back-to-back); see README.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths (all overridable via env) ──────────────────────────────────────────
VENV_PY="${MLX_VENV_PY:-$HOME/src/mlx/.venv-apertus/bin/python}"
MLX_VLM_SRC="${MLX_VLM_SRC:-$HOME/src/mlx-vlm-apertus}"
MODELS_ROOT="${MLX_MODELS:-$HOME/src/mlx/models}"
HF_DIR="${HF_DIR:-$MODELS_ROOT/Apertus-v1.5-8B}"
MODEL_DIR="${MODEL_DIR:-$MODELS_ROOT/Apertus-v1.5-8B-MLX-8bit-omni}"
MODEL_ALIAS="${MODEL_ALIAS:-Apertus-v1.5-8B-local}"
STATE_DIR="${STATE_DIR:-$HOME/.mlx-apertus}"
LOG_FILE="${LOG_FILE:-$STATE_DIR/logs/server.log}"

BIND_HOST="${BIND_HOST:-127.0.0.1}"
PORT="${PORT:-8890}"

# ── Server parameters ────────────────────────────────────────────────────────
PREFILL_STEP="${PREFILL_STEP:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
LOG_PROGRESS="${LOG_PROGRESS:-10}"
KV_BITS="${KV_BITS:-}"            # empty = f16
_MEM_PROBE_INTERVAL="${MEM_PROBE_INTERVAL:-5}"
# Run next to another mlx server only if you know both fit (see MEMORY).
ALLOW_COEXIST="${ALLOW_COEXIST:-0}"

# APC. The cache key includes image content and audio features, so two
# prompts that differ only in their image or audio do not share entries.
ENABLE_APC="${ENABLE_APC:-1}"
APC_ENTRIES="${APC_ENTRIES:-2}"
APC_DISK="${APC_DISK:-$STATE_DIR/apc}"
APC_DISK_MAX_GB="${APC_DISK_MAX_GB:-20}"
APC_MIN_FREE_RAM_GB="${APC_MIN_FREE_RAM_GB:-3.0}"

# ── Checks ───────────────────────────────────────────────────────────────────
[[ -x "$VENV_PY" ]] || {
  echo "ERROR: no venv python at $VENV_PY" >&2
  echo "       uv venv --python 3.12 ${VENV_PY:h:h}" >&2
  echo "       VIRTUAL_ENV=${VENV_PY:h:h} uv pip install -e $MLX_VLM_SRC" >&2
  exit 1
}
"$VENV_PY" -c "import mlx_vlm.models.apertus1p5.audio" 2>/dev/null || {
  echo "ERROR: mlx_vlm in $VENV_PY has no apertus1p5 with audio." >&2
  echo "       Install the apertus1p5 branch: VIRTUAL_ENV=${VENV_PY:h:h} uv pip install -e $MLX_VLM_SRC" >&2
  exit 1
}
[[ -f "$MODEL_DIR/config.json" ]] || {
  echo "ERROR: no checkpoint at $MODEL_DIR" >&2
  echo "       Request access at https://huggingface.co/swiss-ai/Apertus-v1.5-8B, download it" >&2
  echo "       to $HF_DIR (HF_HUB_DISABLE_XET=1 hf download ...), then convert:" >&2
  echo "       $VENV_PY -m mlx_vlm convert --hf-path $HF_DIR --mlx-path $MODEL_DIR \\" >&2
  echo "           -q --q-bits 8 --q-group-size 64 --dtype bfloat16" >&2
  exit 1
}

_WIRED_MB=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
if (( _WIRED_MB > 40960 )); then
  echo "WARNING: iogpu.wired_limit_mb = ${_WIRED_MB}; 45056 panicked this machine before." >&2
fi

if lsof -iTCP:$PORT -sTCP:LISTEN -n &>/dev/null; then
  echo "ERROR: port $PORT is in use." >&2
  exit 1
fi
if [[ "$ALLOW_COEXIST" != "1" ]] && pgrep -f "mlx_vlm.server|mlx_vlm server" >/dev/null; then
  echo "ERROR: another mlx-vlm server is running (Kolibri alone needs ~35 GB)." >&2
  echo "       Stop it, or start with ALLOW_COEXIST=1 if you are sure both fit." >&2
  exit 1
fi

mkdir -p "${LOG_FILE:h}"
ln -sfn "$MODEL_DIR" "$MODELS_ROOT/$MODEL_ALIAS"

MLX_VER=$("$VENV_PY" -c "import importlib.metadata as m;print(m.version('mlx'))" 2>/dev/null || echo "?")
VLM_REV=$(git -C "$MLX_VLM_SRC" rev-parse --short HEAD 2>/dev/null || echo "?")

echo "Apertus 1.5 ($MODEL_ALIAS) · mlx-vlm apertus1p5@$VLM_REV · mlx $MLX_VER"
echo "  model   : $MODEL_ALIAS -> $MODEL_DIR"
echo "  server  : http://$BIND_HOST:$PORT/v1   (log: $LOG_FILE)"
echo "  wired   : ${_WIRED_MB} MB · prefill step $PREFILL_STEP · seqs $MAX_NUM_SEQS · KV ${KV_BITS:-f16}"
echo "  APC     : $([[ "$ENABLE_APC" == "1" ]] && echo "on, $APC_ENTRIES entries, disk $APC_DISK (${APC_DISK_MAX_GB} GB)" || echo off)"

args=(
  --host                  "$BIND_HOST"
  --port                  "$PORT"
  --model                 "$MODEL_ALIAS"
  --prefill-step-size     "$PREFILL_STEP"
  --max-num-seqs          "$MAX_NUM_SEQS"
  --log-progress-interval "$LOG_PROGRESS"
  --thinking-start-token  "<|inner_prefix|>"
  --thinking-end-token    "<|inner_suffix|>"
)
[[ -n "$KV_BITS" ]] && args+=( --kv-bits "$KV_BITS" )

if [[ "$ENABLE_APC" == "1" ]]; then
  export APC_ENABLED=1
  export APC_EXACT_CACHE_ENTRIES="$APC_ENTRIES"
  if [[ -n "$APC_DISK" ]]; then
    mkdir -p "$APC_DISK"
    export APC_DISK_PATH="$APC_DISK"
    export APC_DISK_MAX_GB="$APC_DISK_MAX_GB"
    export APC_DISK_MIN_FREE_RAM_GB="$APC_MIN_FREE_RAM_GB"
  fi
fi

# CWD = models root so the relative alias name resolves.
cd "$MODELS_ROOT"

# Memory sampler, same as in start-mlx_qwen3.8.sh (see the reasoning there):
# Metal buffers do not show in RSS, so sample mlx's own counters from a thread.
_MEM_BOOT=$(cat <<'PY'
import os, sys, threading, time, logging, runpy
import mlx.core as mx

_GIB = 1024 ** 3


def _working_set():
    try:
        info = mx.device_info()
    except AttributeError:
        info = mx.metal.device_info()
    return float(info.get("max_recommended_working_set_size") or 0)


def _sample(interval, ws):
    log = logging.getLogger("memprobe")
    time.sleep(interval)
    while True:
        try:
            active, cache = mx.get_active_memory(), mx.get_cache_memory()
            total = active + cache
            pct = (total / ws * 100.0) if ws else 0.0
            msg = ("mem active=%.2f cache=%.2f sum=%.2f GiB "
                   "(%.0f%% of %.2f GiB working set) peak=%.2f GiB")
            argv = (active / _GIB, cache / _GIB, total / _GIB,
                    pct, ws / _GIB, mx.get_peak_memory() / _GIB)
            log.warning(msg, *argv) if pct >= 85.0 else log.info(msg, *argv)
        except Exception:
            pass
        time.sleep(interval)


_iv = float(os.environ.get("MEM_PROBE_INTERVAL", "5") or 0)
if _iv > 0:
    threading.Thread(target=_sample, args=(_iv, _working_set()), daemon=True).start()

sys.argv[0] = "mlx_vlm.server"
runpy.run_module("mlx_vlm.server", run_name="__main__")
PY
)

if [[ "$_MEM_PROBE_INTERVAL" != "0" ]]; then
  export MEM_PROBE_INTERVAL="$_MEM_PROBE_INTERVAL"
  exec "$VENV_PY" -c "$_MEM_BOOT" "${args[@]}" > >(tee -a "$LOG_FILE") 2>&1
else
  exec "$VENV_PY" -m mlx_vlm.server "${args[@]}" > >(tee -a "$LOG_FILE") 2>&1
fi
