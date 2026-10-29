#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# mlx-vlm reranker server  -  BAAI bge-reranker-v2-m3 (XLM-RoBERTa cross-encoder)
# TARGET HARDWARE:  Apple Silicon, any size (~2.2 GiB, fp32 weights)
#
# Purpose: Aletora's `ranker` role. Multilingual, German included.
# Checkpoint: the original BAAI repo (model.safetensors), no conversion:
#   ./download-mlx-model.sh BAAI/bge-reranker-v2-m3 ~/src/mlx/models/bge-reranker-v2-m3
#
# ── API ─────────────────────────────────────────────────────────────────────
#   POST /v1/rerank {"model": "$MODEL_ALIAS", "query": "...", "documents": [...],
#                    "top_n": N, "return_documents": false}
#   -> {"results": [{"index": i, "relevance_score": s}, ...]}, best first.
#   Same shape as Aletora's OmlxClient.rerank, which posts <base>/rerank.
#   Aletora: provider kind omlx, base http://host.docker.internal:8893/v1.
#   Scores are sigmoid, 0..1. Measured 2026-10-28: relevant 0.981, others 0.0.
#   The colima VM reaches the 127.0.0.1 bind; 0.0.0.0 is not needed.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

VENV_PY="${MLX_VENV_PY:-$HOME/src/mlx/.venv/bin/python}"
MODELS_ROOT="${MLX_MODELS:-$HOME/src/mlx/models}"
MODEL_DIR="${MODEL_DIR:-$MODELS_ROOT/bge-reranker-v2-m3}"
MODEL_ALIAS="${MODEL_ALIAS:-bge-reranker-v2-m3-local}"
STATE_DIR="${STATE_DIR:-$HOME/.mlx-rerank}"
LOG_FILE="${LOG_FILE:-$STATE_DIR/logs/server.log}"

BIND_HOST="${BIND_HOST:-127.0.0.1}"
PORT="${PORT:-8893}"
# MLX keeps freed buffers for reuse, unbounded by default. Measured 2026-10-28:
# 11 GB footprint for 2.3 GB of weights; under the resulting swap one rerank
# of ~4200 tokens went from 0.3 s to 10 s. 0 = no cache.
CACHE_LIMIT_MB="${CACHE_LIMIT_MB:-512}"

[[ -x "$VENV_PY" ]] || {
  echo "ERROR: no venv python at $VENV_PY -- run ./install-prereqs.sh" >&2
  exit 1
}
[[ -f "$MODEL_DIR/config.json" ]] || {
  echo "ERROR: no checkpoint at $MODEL_DIR" >&2
  echo "       ./download-mlx-model.sh BAAI/bge-reranker-v2-m3 $MODEL_DIR" >&2
  exit 1
}
if lsof -iTCP:$PORT -sTCP:LISTEN -n &>/dev/null; then
  echo "ERROR: port $PORT is in use." >&2
  exit 1
fi

mkdir -p "${LOG_FILE:h}"
ln -sfn "$MODEL_DIR" "$MODELS_ROOT/$MODEL_ALIAS"

echo "bge-reranker-v2-m3 ($MODEL_ALIAS) -> $MODEL_DIR"
echo "  server  : http://$BIND_HOST:$PORT/v1/rerank   (log: $LOG_FILE)"
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
  --reranker-model "$MODEL_ALIAS" \
  > >(tee -a "$LOG_FILE") 2>&1
