#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# Capability column (sovereign-models): MMLU 5-shot of one model at 8 bit.
#
#   ./mmlu-eu.sh <id> <original hf repo>
#
# Converts the original to 8 bit (mlx_lm.convert, affine, group size 64; the
# same recipe for every model, so the numbers compare), runs measure-mmlu.py
# on it, and deletes the download and the conversion again. The result is in
# ~/src/mlx/mmlu5/runs/<id>-8bit.jsonl. Only for models whose 8-bit weights
# fit in memory (about 32B parameters on a 48 GB Mac).
# ─────────────────────────────────────────────────────────────────────────────
set -u
ID=$1; REPO=$2
HERE=${0:A:h}; PY=~/src/mlx/.venv/bin/python
OUT=~/src/mlx/models/eu/$ID-8bit
CACHE=~/.cache/huggingface/hub/models--${REPO:gs/\//--}
log() { print -r -- "[$(date +%H:%M:%S)] $ID: $*"; }

if [[ -f ~/src/mlx/mmlu5/runs/$ID-8bit.jsonl ]]; then log "already measured"; exit 0; fi
export HF_HUB_DISABLE_XET=1
log "convert $REPO to 8 bit"
$PY -m mlx_lm convert --hf-path $REPO --mlx-path $OUT -q --q-bits 8 --q-group-size 64 > /tmp/mmlu-eu-$ID.log 2>&1 \
  || { log "conversion failed: $(tail -1 /tmp/mmlu-eu-$ID.log)"; rm -rf $OUT $CACHE; exit 1; }
rm -rf $CACHE
log "mmlu"
$PY $HERE/measure-mmlu.py run --model $OUT --arm $ID-8bit --loader lm 2>&1 | tail -1
rm -rf $OUT
log "done"
