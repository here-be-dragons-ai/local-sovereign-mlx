#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# mlx-vlm server start script  -  Aleph Alpha Kolibri 1 (78B-A3.5B MoE)
# TARGET HARDWARE:  Apple Silicon, 48 GB, iogpu.wired_limit_mb=40960
#
# Checkpoint: Kolibri-1-MLX-3bit, made by convert-kolibri.py from the FP8
# release -- routed experts 3 bit, attention / shared expert / embedding /
# lm_head 6 bit, router bf16. 3.61 bits per weight, 32.8 GiB active after load.
# Needs patch 0050 (mlx_vlm/models/kolibri1) until upstream carries the model.
#
# ── MEMORY ──────────────────────────────────────────────────────────────────
#   weights        32.8 GiB
#   KV cache       ~20 KiB/token in f16: only the 10 full-attention layers grow
#                  with the context (4 KV heads x 128 x 2 x 2 B each); the 40
#                  sliding-window layers hold 513 tokens each, whatever the
#                  context length.
#   Measured 2026-10-03, M5 Pro: peak 35.3 GB on short prompts.
#   It does NOT fit next to the Qwen3.8 server; this script refuses to start
#   while the port is taken.
#
# ── SPEED (measured 2026-10-03, M5 Pro, mlx-vlm 0.7.4, mlx 0.32.2) ──────────
#   decode    ~70 t/s short context, 57 t/s at 23k, 40 t/s at 96k
#   prefill   ~1570 t/s at 23k, ~1020 t/s at 96k
#   No drafter: none exists for this model, and at 3.5B active parameters
#   decode is not the bottleneck.
#
# ── REASONING ───────────────────────────────────────────────────────────────
#   Clients choose per request with the TOP-LEVEL fields `reasoning_effort`
#   (none / low / medium / high) or `enable_thinking`; the server ignores
#   them inside chat_template_kwargs. A request with neither does not think.
#   The thinking comes back in `reasoning`, separate from `content`.
#
# ── TOOLS ───────────────────────────────────────────────────────────────────
#   The template is matched to the `json_tools` parser (<tool_call> + JSON).
#   Verified through the server: single, nested/boolean arguments, two calls
#   in one turn, tool-result round trip, streaming. Unprompted, the model
#   prefers one call per turn over parallel calls.
#
# Prerequisites: install-prereqs.sh has run, patches/apply-patches.sh has
# applied 0050, the checkpoint exists (see convert-kolibri.py).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths (all overridable via env) ──────────────────────────────────────────
VENV_PY="${MLX_VENV_PY:-$HOME/src/mlx/.venv/bin/python}"
MODELS_ROOT="${MLX_MODELS:-$HOME/src/mlx/models}"
MODEL_DIR="${MODEL_DIR:-$MODELS_ROOT/Kolibri-1-MLX-3bit}"
MODEL_ALIAS="${MODEL_ALIAS:-Kolibri-1-local}"
STATE_DIR="${STATE_DIR:-$HOME/.mlx-kolibri}"
LOG_FILE="${LOG_FILE:-$STATE_DIR/logs/server.log}"

BIND_HOST="${BIND_HOST:-127.0.0.1}"
PORT="${PORT:-8888}"

# ── Server parameters ────────────────────────────────────────────────────────
PREFILL_STEP="${PREFILL_STEP:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
LOG_PROGRESS="${LOG_PROGRESS:-10}"
KV_BITS="${KV_BITS:-}"            # empty = f16; the KV cache is small anyway
_MEM_PROBE_INTERVAL="${MEM_PROBE_INTERVAL:-5}"

# APC. Kolibri caches are mixed: KVCache in the full-attention layers,
# RotatingKVCache in the sliding-window layers; the exact mode handles both
# (self-check: pageable / windowed). Measured 2026-10-03 on a 15.5k-token
# conversation: cold 20.3 s, follow-up turn 1.5 s (14,336 tokens restored),
# identical repeat 0.2 s, answers unchanged.
ENABLE_APC="${ENABLE_APC:-1}"
APC_ENTRIES="${APC_ENTRIES:-2}"
APC_DISK="${APC_DISK:-$STATE_DIR/apc}"
APC_DISK_MAX_GB="${APC_DISK_MAX_GB:-40}"
APC_MIN_FREE_RAM_GB="${APC_MIN_FREE_RAM_GB:-3.0}"

# ── Checks ───────────────────────────────────────────────────────────────────
[[ -x "$VENV_PY" ]] || { echo "ERROR: no venv python at $VENV_PY (install-prereqs.sh)" >&2; exit 1; }
[[ -f "$MODEL_DIR/config.json" ]] || {
  echo "ERROR: no checkpoint at $MODEL_DIR" >&2
  echo "       ./convert-kolibri.py \$MLX_MODELS/Kolibri-1-FP8 $MODEL_DIR" >&2
  exit 1
}
"$VENV_PY" -c "import mlx_vlm.models.kolibri1" 2>/dev/null || {
  echo "ERROR: mlx_vlm has no kolibri1 model -- patch 0050 missing." >&2
  echo "       MLX_VENV_PY=$VENV_PY ./patches/apply-patches.sh" >&2
  exit 1
}

_WIRED_MB=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
if (( _WIRED_MB < 40960 )); then
  echo "ERROR: iogpu.wired_limit_mb = ${_WIRED_MB} (< 40960)." >&2
  echo "       The weights alone are 32.8 GiB; the macOS default working set" >&2
  echo "       (2/3 of 48 GB = 32 GiB) cannot hold them." >&2
  echo "       sudo sysctl -w iogpu.wired_limit_mb=40960   (not higher, see README)" >&2
  exit 1
fi

if lsof -iTCP:$PORT -sTCP:LISTEN -n &>/dev/null; then
  echo "ERROR: port $PORT is in use -- is the Qwen3.8 server running?" >&2
  echo "       Both do not fit into memory at once; stop it first." >&2
  exit 1
fi

mkdir -p "${LOG_FILE:h}"
ln -sfn "$MODEL_DIR" "$MODELS_ROOT/$MODEL_ALIAS"

MLX_VER=$("$VENV_PY" -c "import importlib.metadata as m;print(m.version('mlx'))" 2>/dev/null || echo "?")
VLM_VER=$("$VENV_PY" -c "import importlib.metadata as m;print(m.version('mlx-vlm'))" 2>/dev/null || echo "?")

echo "Kolibri 1 · mlx-vlm $VLM_VER · mlx $MLX_VER"
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
