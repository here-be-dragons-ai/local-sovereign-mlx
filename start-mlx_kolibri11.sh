#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# mlx-vlm server start script  -  Aleph Alpha Kolibri 1.1 (75B-A3.4B MoE)
# TARGET HARDWARE:  Apple Silicon, 48 GB, iogpu.wired_limit_mb=40960
#
# Checkpoint: Kolibri-1.1-MLX-3bit, made by convert-kolibri.py from the FP8
# release -- routed experts 3 bit, attention / shared expert / dense MLP /
# embedding / lm_head 6 bit, router bf16. 3.61 bits per weight, 31.6 GiB.
# mlx-vlm has no kolibri1_1. The build brings its model code (kolibri1_1.py,
# "model_file" in config.json; from kolibri1_1/ in this repository), which
# mlx-vlm 0.8.0 loads as it is. See kolibri1_1/language.py for what differs
# from Kolibri 1: sparse attention, K/V sharing, a dense prefix.
# While loading, the log shows one line
#   ERROR - Model type kolibri1_1 not supported. Error: No module named ...
# That is mlx-vlm asking its own model registry for an image processor
# (utils.load_image_processor); it catches the error and goes on without one,
# which is right for a text model.
#
# ── MEMORY ──────────────────────────────────────────────────────────────────
#   weights        31.6 GiB
#   KV cache       ~8 KiB/token in f16: 4 of the 10 sparse layers own K/V (the
#                  others read theirs), 2 KiB each; the 40 sliding-window
#                  layers hold 513 tokens each. 0.95 GiB at 96k tokens.
#   Measured 2026-11-24, M5 Pro, through this server (measure-kolibri-ttft.py,
#   one run): peak 32.5 GiB at 1k tokens, 35.1 at 8k, 37.6 at 32k, 38.0 at
#   64k and 96k.
#   It does NOT fit next to another model server; this script refuses to
#   start while the port is taken.
#
# ── SPEED (measured 2026-11-24, M5 Pro, mlx-vlm 0.8.0, mlx 0.32.4) ──────────
#   decode    in process: ~65 t/s short context, 47 t/s at 16k, 32 t/s at 96k
#   first token, cold (this server, one run): 0.8 s at 1k, 5.1 s at 8k, 31 s
#             at 32k, 107 s at 64k, 206 s at 96k (468 t/s); Kolibri 1: 0.8 /
#             5.3 / 24 / 59 / 109 s
#   first token on an exact-APC hit: 0.2 / 0.3 / 0.4 / 0.6 / 1.1 s
#   Past ~2,200 tokens every sparse layer scores all keys of the context for
#   every query and selects 2048 of them; that is the cost at long context.
#   In prefill the attention still runs over all keys under a mask, so
#   sparsity saves nothing there.
#   No drafter: none exists for this model.
#
# ── REASONING, TOOLS ────────────────────────────────────────────────────────
#   As for Kolibri 1 (start-mlx_kolibri.sh): same tokenizer, same chat
#   template. `reasoning_effort` (none / low / medium / high) or
#   `enable_thinking` on the TOP LEVEL of the request.
#
# Prerequisites: install-prereqs.sh has run (mlx-vlm 0.8.0 or later), the
# checkpoint exists (convert-kolibri.py builds it from the FP8 release).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths (all overridable via env) ──────────────────────────────────────────
VENV_PY="${MLX_VENV_PY:-$HOME/src/mlx/.venv/bin/python}"
MODELS_ROOT="${MLX_MODELS:-$HOME/src/mlx/models}"
MODEL_DIR="${MODEL_DIR:-$MODELS_ROOT/Kolibri-1.1-MLX-3bit}"
MODEL_ALIAS="${MODEL_ALIAS:-Kolibri-1.1-local}"
STATE_DIR="${STATE_DIR:-$HOME/.mlx-kolibri11}"
LOG_FILE="${LOG_FILE:-$STATE_DIR/logs/server.log}"

BIND_HOST="${BIND_HOST:-127.0.0.1}"
PORT="${PORT:-8888}"

# ── Server parameters ────────────────────────────────────────────────────────
PREFILL_STEP="${PREFILL_STEP:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
LOG_PROGRESS="${LOG_PROGRESS:-10}"
KV_BITS="${KV_BITS:-}"            # empty = f16; the KV cache is small anyway
_MEM_PROBE_INTERVAL="${MEM_PROBE_INTERVAL:-5}"

# APC. Kolibri caches are mixed: KVCache in the sparse layers that own K/V,
# RotatingKVCache in the sliding-window layers; the exact mode handles both
# (self-check: pageable / windowed).
#
# APC_RESERVE_GB. APC keeps a snapshot only while the headroom -- the smaller
# of (working set - active memory) and (free + inactive RAM per vm_stat + mlx
# cache) -- covers this reserve plus the prefill reserve plus the snapshot.
# mlx-vlm's automatic reserve is a tenth of the working set, 4 GiB here; next
# to 32 GB of weights little RAM is free. Measured on Kolibri 1 (2026-10-08):
# with the automatic reserve follow-up turns from ~32k tokens up missed the
# cache, with 1.5 they hit. The working-set term still guards the Metal limit.
# With reasoning_effort=none, follow-up turns miss the final snapshot anyway:
# the template's empty <think></think> is not rendered for earlier turns.
ENABLE_APC="${ENABLE_APC:-1}"
APC_RESERVE_GB="${APC_RESERVE_GB:-1.5}"
APC_ENTRIES="${APC_ENTRIES:-2}"
APC_DISK="${APC_DISK:-$STATE_DIR/apc}"
APC_DISK_MAX_GB="${APC_DISK_MAX_GB:-40}"
APC_MIN_FREE_RAM_GB="${APC_MIN_FREE_RAM_GB:-3.0}"

# ── Checks ───────────────────────────────────────────────────────────────────
[[ -x "$VENV_PY" ]] || { echo "ERROR: no venv python at $VENV_PY (install-prereqs.sh)" >&2; exit 1; }
[[ -f "$MODEL_DIR/config.json" ]] || {
  echo "ERROR: no checkpoint at $MODEL_DIR" >&2
  echo "       ./convert-kolibri.py \$MLX_MODELS/Kolibri-1.1-FP8 $MODEL_DIR" >&2
  exit 1
}
_MODEL_FILE=$("$VENV_PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('model_file',''))" "$MODEL_DIR/config.json")
[[ -n "$_MODEL_FILE" && -f "$MODEL_DIR/$_MODEL_FILE" ]] || {
  echo "ERROR: $MODEL_DIR brings no model code (model_file in config.json)." >&2
  echo "       ./convert-kolibri.py --code-only \$MLX_MODELS/Kolibri-1.1-FP8 $MODEL_DIR" >&2
  exit 1
}
"$VENV_PY" -c "import inspect, mlx_vlm.utils as u; assert 'model_file' in inspect.getsource(u.get_model_and_args)" 2>/dev/null || {
  echo "ERROR: this mlx_vlm does not load model code from a checkpoint (needs 0.8.0)." >&2
  echo "       ./install-prereqs.sh" >&2
  exit 1
}

_WIRED_MB=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
if (( _WIRED_MB < 40960 )); then
  echo "ERROR: iogpu.wired_limit_mb = ${_WIRED_MB} (< 40960)." >&2
  echo "       The weights alone are 31.6 GiB; the macOS default working set" >&2
  echo "       (2/3 of 48 GB = 32 GiB) leaves nothing for the context." >&2
  echo "       sudo sysctl -w iogpu.wired_limit_mb=40960   (not higher, see README)" >&2
  exit 1
fi

if lsof -iTCP:$PORT -sTCP:LISTEN -n &>/dev/null; then
  echo "ERROR: port $PORT is in use -- is another model server running?" >&2
  echo "       Both do not fit into memory at once; stop it first." >&2
  exit 1
fi

mkdir -p "${LOG_FILE:h}"
ln -sfn "$MODEL_DIR" "$MODELS_ROOT/$MODEL_ALIAS"

MLX_VER=$("$VENV_PY" -c "import importlib.metadata as m;print(m.version('mlx'))" 2>/dev/null || echo "?")
VLM_VER=$("$VENV_PY" -c "import importlib.metadata as m;print(m.version('mlx-vlm'))" 2>/dev/null || echo "?")

echo "Kolibri 1.1 · mlx-vlm $VLM_VER · mlx $MLX_VER"
echo "  model   : $MODEL_ALIAS -> $MODEL_DIR"
echo "  server  : http://$BIND_HOST:$PORT/v1   (log: $LOG_FILE)"
echo "  wired   : ${_WIRED_MB} MB · prefill step $PREFILL_STEP · seqs $MAX_NUM_SEQS · KV ${KV_BITS:-f16}"
echo "  note    : one \"ERROR - Model type kolibri1_1 not supported\" while loading is expected (no image processor)"
echo "  APC     : $([[ "$ENABLE_APC" == "1" ]] && echo "on, $APC_ENTRIES entries, reserve ${APC_RESERVE_GB} GB, disk $APC_DISK (${APC_DISK_MAX_GB} GB)" || echo off)"

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
  export APC_MEMORY_RESERVE_GB="$APC_RESERVE_GB"
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
