#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# mlx-vlm server start script  -  Qwen3.6-35B-A3B (MoE, text + image, MLX 4bit)
# TARGET HARDWARE:  Apple Silicon, 48 GB
#
# Purpose: the generation model for Aletora (answer, answer_light, judge, ocr,
# diagram). Runs NEXT TO start-mlx_embed.sh and start-mlx_rerank.sh, not next
# to Qwen3.8 / Kolibri / Apertus.
#
# Checkpoint: mlx-community/Qwen3.6-35B-A3B-4bit, 4bit affine (group size 64),
# router gates 8 bit. 20.4 GB on disk.
#   ./download-mlx-model.sh mlx-community/Qwen3.6-35B-A3B-4bit \
#       ~/src/mlx/models/Qwen3.6-35B-A3B-MLX-4bit
#
# ── MEMORY (computed from config.json, not measured) ────────────────────────
#   weights        ~19.0 GiB
#   KV cache       20 KiB/token in f16 (10 full-attn layers x 2 KV heads x 256 x 2 x 2 B)
#                  32k context = 640 MiB per sequence
#   GDN state      ~60 MiB per sequence (30 linear layers x 32 v-heads x 128 x 128 x 4 B)
#   Budget on 48 GB with Aletora: 8 GiB macOS + 8 GiB colima + ~24 GiB models
#   (this + embed + rerank). Leaves ~8 GiB.
#
# ── REASONING ───────────────────────────────────────────────────────────────
#   Off. The server is started WITHOUT --enable-thinking, so a request that
#   does not set enable_thinking gets the template's pre-closed <think></think>.
#   Clients may still turn it on per request (enable_thinking: true or
#   reasoning_effort low|medium|high). THINKING_BUDGET caps that case.
#
# ── ALETORA ─────────────────────────────────────────────────────────────────
#   Containers under colima reach the host as host.docker.internal. Provider
#   base: http://host.docker.internal:8889/v1, model: $MODEL_ALIAS.
#   The colima VM reaches the 127.0.0.1 bind (measured 2026-10-28 on the embed
#   and rerank servers); 0.0.0.0 is not needed.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths (all overridable via env) ──────────────────────────────────────────
VENV_PY="${MLX_VENV_PY:-$HOME/src/mlx/.venv/bin/python}"
MODELS_ROOT="${MLX_MODELS:-$HOME/src/mlx/models}"
MODEL_DIR="${MODEL_DIR:-$MODELS_ROOT/Qwen3.6-35B-A3B-MLX-4bit}"
# mlx-vlm has no --alias: the request's model string is the load path. Symlink
# with the alias name + relative preload, same as start-mlx_qwen3.8.sh.
MODEL_ALIAS="${MODEL_ALIAS:-Qwen3.6-35B-A3B-local}"
STATE_DIR="${STATE_DIR:-$HOME/.mlx-qwen36}"
LOG_FILE="${LOG_FILE:-$STATE_DIR/logs/server.log}"

# Not $HOST: zsh sets it to the host name.
BIND_HOST="${BIND_HOST:-127.0.0.1}"
PORT="${PORT:-8889}"

# ── Server parameters ────────────────────────────────────────────────────────
PREFILL_STEP="${PREFILL_STEP:-2048}"
# Aletora runs chat and ingest workers in parallel. 4 x 32k = 2.5 GiB KV.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
# OCR pages are unique images; caching their features buys nothing.
VISION_CACHE="${VISION_CACHE:-2}"
LOG_PROGRESS="${LOG_PROGRESS:-10}"
KV_BITS="${KV_BITS:-}"            # empty = f16
# Only applies when a client turns thinking on.
THINKING_BUDGET="${THINKING_BUDGET:-1024}"
_MEM_PROBE_INTERVAL="${MEM_PROBE_INTERVAL:-5}"

# APC. RAG prompts share the system prompt; follow-up turns share the history.
ENABLE_APC="${ENABLE_APC:-1}"
APC_ENTRIES="${APC_ENTRIES:-4}"
APC_DISK="${APC_DISK:-$STATE_DIR/apc}"
APC_DISK_MAX_GB="${APC_DISK_MAX_GB:-20}"
APC_MIN_FREE_RAM_GB="${APC_MIN_FREE_RAM_GB:-3.0}"

# Warmup. The first large prefill after load is cold: measured 2026-10-28,
# 4466 tokens at 291 tok/s (15.3 s), the same server ~2500 tok/s a minute
# later. Short probes do not warm it -- the PREFILL_STEP-sized chunks have to
# run once. So one throwaway prompt of > PREFILL_STEP tokens is sent as soon as
# the server answers. Costs one APC slot until it is evicted.
WARMUP="${WARMUP:-1}"
WARMUP_TOKENS="${WARMUP_TOKENS:-4500}"

# ── Checks ───────────────────────────────────────────────────────────────────
[[ -x "$VENV_PY" ]] || {
  echo "ERROR: no venv python at $VENV_PY -- run ./install-prereqs.sh" >&2
  exit 1
}
[[ -f "$MODEL_DIR/config.json" ]] || {
  echo "ERROR: no checkpoint at $MODEL_DIR" >&2
  echo "       ./download-mlx-model.sh mlx-community/Qwen3.6-35B-A3B-4bit $MODEL_DIR" >&2
  exit 1
}
if [[ -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
  MISSING=$(python3 -c "
import json,os
d=json.load(open('$MODEL_DIR/model.safetensors.index.json'))
print(' '.join(f for f in sorted(set(d['weight_map'].values())) if not os.path.exists(os.path.join('$MODEL_DIR',f))))")
  [[ -z "$MISSING" ]] || {
    echo "ERROR: model incomplete -- missing shards: $MISSING" >&2
    echo "       ./download-mlx-model.sh mlx-community/Qwen3.6-35B-A3B-4bit $MODEL_DIR" >&2
    exit 1
  }
fi

_WIRED_MB=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
if (( _WIRED_MB > 40960 )); then
  echo "WARNING: iogpu.wired_limit_mb = ${_WIRED_MB}; 45056 panicked this machine before." >&2
fi

if lsof -iTCP:$PORT -sTCP:LISTEN -n &>/dev/null; then
  echo "ERROR: port $PORT is in use." >&2
  exit 1
fi
# Only a second GENERATION server is a conflict. Embed/rerank servers start
# with --embedding-model / --reranker-model and no " --model ".
if [[ "${ALLOW_COEXIST:-0}" != "1" ]] && pgrep -f -- "mlx_vlm.* --model " >/dev/null; then
  echo "ERROR: another mlx-vlm generation server is running (Qwen3.8/Kolibri/Apertus)." >&2
  echo "       Together they do not fit. Stop it, or ALLOW_COEXIST=1." >&2
  exit 1
fi

mkdir -p "${LOG_FILE:h}"
ln -sfn "$MODEL_DIR" "$MODELS_ROOT/$MODEL_ALIAS"

MLX_VER=$("$VENV_PY" -c "import importlib.metadata as m;print(m.version('mlx'))" 2>/dev/null || echo "?")
VLM_VER=$("$VENV_PY" -c "import importlib.metadata as m;print(m.version('mlx-vlm'))" 2>/dev/null || echo "?")

echo "Qwen3.6-35B-A3B ($MODEL_ALIAS) · mlx-vlm $VLM_VER · mlx $MLX_VER"
echo "  model   : $MODEL_ALIAS -> $MODEL_DIR"
echo "  server  : http://$BIND_HOST:$PORT/v1   (log: $LOG_FILE)"
echo "  wired   : ${_WIRED_MB} MB · prefill step $PREFILL_STEP · seqs $MAX_NUM_SEQS · KV ${KV_BITS:-f16}"
echo "  thinking: off by default (budget $THINKING_BUDGET when a client enables it)"
echo "  APC     : $([[ "$ENABLE_APC" == "1" ]] && echo "on, $APC_ENTRIES entries, disk $APC_DISK (${APC_DISK_MAX_GB} GB)" || echo off)"
echo "  warmup  : $([[ "$WARMUP" == "1" ]] && echo "on, ~$WARMUP_TOKENS tokens after start" || echo off)"

args=(
  --host                  "$BIND_HOST"
  --port                  "$PORT"
  --model                 "$MODEL_ALIAS"
  --prefill-step-size     "$PREFILL_STEP"
  --max-num-seqs          "$MAX_NUM_SEQS"
  --vision-cache-size     "$VISION_CACHE"
  --log-progress-interval "$LOG_PROGRESS"
  --thinking-budget       "$THINKING_BUDGET"
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

# Background warmup: waits for the server, sends one long prompt, logs the
# result. Detached (&!) so it survives the exec below.
if [[ "$WARMUP" == "1" ]]; then
  _WARM_HOST="$BIND_HOST"
  [[ "$_WARM_HOST" == "0.0.0.0" ]] && _WARM_HOST=127.0.0.1
  "$VENV_PY" - "http://$_WARM_HOST:$PORT/v1" "$MODEL_ALIAS" "$WARMUP_TOKENS" \
      >> "$LOG_FILE" 2>&1 <<'PY' &!
import json, sys, time, urllib.request

base, model, tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
deadline = time.time() + 900
while time.time() < deadline:
    try:
        urllib.request.urlopen(f"{base}/models", timeout=5)
        break
    except Exception:
        time.sleep(2)
else:
    print("warmup: server not ready after 900 s, skipped", flush=True)
    sys.exit(0)

# ~1.3 tokens per word for German prose; unique per start so APC never hits it.
words = ("Die Ordination meldet der Kammer jede Änderung der Vertragsdaten "
         "schriftlich und fristgerecht innerhalb von vier Wochen").split()
n = int(tokens / 1.3)
text = f"{time.time()} " + " ".join(words[i % len(words)] for i in range(n))
body = {"model": model, "max_tokens": 1, "temperature": 0,
        "messages": [{"role": "user", "content": text}]}
req = urllib.request.Request(f"{base}/chat/completions", json.dumps(body).encode(),
                             {"content-type": "application/json"})
started = time.time()
try:
    usage = json.load(urllib.request.urlopen(req, timeout=600)).get("usage", {})
    print(f"warmup: {usage.get('prompt_tokens')} prompt tokens in "
          f"{time.time() - started:.1f} s", flush=True)
except Exception as exc:
    print(f"warmup: failed: {exc}", flush=True)
PY
fi

# CWD = models root so the relative alias name resolves.
cd "$MODELS_ROOT"

# Memory sampler, same as in start-mlx_qwen3.8.sh: Metal buffers do not show
# in RSS, so sample mlx's own counters from a thread.
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
