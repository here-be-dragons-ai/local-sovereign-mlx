#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# mlx-vlm embedding server  -  BAAI bge-m3 (XLM-RoBERTa, 1024 dim, fp16)
# TARGET HARDWARE:  Apple Silicon, any size (~1.2 GiB)
#
# Purpose: Aletora's `embed` role. Same model as the hosted openrouter
# baai/bge-m3, so local <-> hosted is a routing flip, not a re-index -- once
# cosine parity is checked (4bit/fp16 drift).
#   ./download-mlx-model.sh mlx-community/bge-m3-mlx-fp16 \
#       ~/src/mlx/models/bge-m3-mlx-fp16
#
# ── POOLING ─────────────────────────────────────────────────────────────────
#   bge-m3 dense vectors are the CLS token. The mlx-community repo ships no
#   1_Pooling/config.json, and without it mlx-vlm mean-pools: valid-looking,
#   wrong vectors. The script writes the BAAI file if it is missing.
#
# ── API ─────────────────────────────────────────────────────────────────────
#   POST /v1/embeddings {"model": "$MODEL_ALIAS", "input": [...]}
#   Vectors are L2-normalised. No task prefixes (bge-m3 is symmetric).
#   Inputs are truncated at 8192 tokens. `dimensions` is ignored.
#   Aletora: provider base http://host.docker.internal:8892/v1.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

VENV_PY="${MLX_VENV_PY:-$HOME/src/mlx/.venv/bin/python}"
MODELS_ROOT="${MLX_MODELS:-$HOME/src/mlx/models}"
MODEL_DIR="${MODEL_DIR:-$MODELS_ROOT/bge-m3-mlx-fp16}"
MODEL_ALIAS="${MODEL_ALIAS:-bge-m3-local}"
STATE_DIR="${STATE_DIR:-$HOME/.mlx-embed}"
LOG_FILE="${LOG_FILE:-$STATE_DIR/logs/server.log}"

BIND_HOST="${BIND_HOST:-127.0.0.1}"
PORT="${PORT:-8892}"
# MLX keeps freed buffers for reuse, unbounded by default. Measured 2026-10-28:
# 5.4 GB footprint for 1.1 GB of weights, which pushed the 48 GB machine into
# swap and made the next request page the weights back in. 0 = no cache.
CACHE_LIMIT_MB="${CACHE_LIMIT_MB:-512}"

[[ -x "$VENV_PY" ]] || {
  echo "ERROR: no venv python at $VENV_PY -- run ./install-prereqs.sh" >&2
  exit 1
}
[[ -f "$MODEL_DIR/config.json" ]] || {
  echo "ERROR: no checkpoint at $MODEL_DIR" >&2
  echo "       ./download-mlx-model.sh mlx-community/bge-m3-mlx-fp16 $MODEL_DIR" >&2
  exit 1
}
if lsof -iTCP:$PORT -sTCP:LISTEN -n &>/dev/null; then
  echo "ERROR: port $PORT is in use." >&2
  exit 1
fi

if [[ ! -f "$MODEL_DIR/1_Pooling/config.json" ]]; then
  mkdir -p "$MODEL_DIR/1_Pooling"
  cat > "$MODEL_DIR/1_Pooling/config.json" <<'JSON'
{
  "word_embedding_dimension": 1024,
  "pooling_mode_cls_token": true,
  "pooling_mode_mean_tokens": false,
  "pooling_mode_max_tokens": false,
  "pooling_mode_mean_sqrt_len_tokens": false
}
JSON
  echo "  wrote $MODEL_DIR/1_Pooling/config.json (CLS pooling)"
fi

mkdir -p "${LOG_FILE:h}"
ln -sfn "$MODEL_DIR" "$MODELS_ROOT/$MODEL_ALIAS"

echo "bge-m3 embeddings ($MODEL_ALIAS) -> $MODEL_DIR"
echo "  server  : http://$BIND_HOST:$PORT/v1/embeddings   (log: $LOG_FILE)"
echo "  cache   : ${CACHE_LIMIT_MB} MB limit"

cd "$MODELS_ROOT"
export CACHE_LIMIT_MB
exec "$VENV_PY" -c '
import os, runpy, sys
import mlx.core as mx
mx.set_cache_limit(int(os.environ["CACHE_LIMIT_MB"]) * 1024 * 1024)
sys.argv[0] = "mlx_vlm.server"
runpy.run_module("mlx_vlm.server", run_name="__main__")
' \
  --host "$BIND_HOST" --port "$PORT" \
  --embedding-model "$MODEL_ALIAS" \
  > >(tee -a "$LOG_FILE") 2>&1
