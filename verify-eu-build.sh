#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# Measure one European model's community MLX build against its original, and
# its MMLU 5-shot at 8 bit (sovereign-models#7 3b, capability column).
#
#   ./verify-eu-build.sh <id> <original hf repo> <build hf repo> <build bits>
#
# 1. download original and build (weights, configs, tokenizer)
# 2. measurement sets with the original's tokenizer and chat template
# 3. check: streamed forward bit-identical to the in-memory forward
# 4. forward original, original in 512-token chunks (noise floor), build;
#    report; one summary line (quality-summary.py) to ~/src/mlx/quality/summaries.jsonl
# 5. MMLU 5-shot on 8 bit: the build if it is 8-bit, else an 8-bit conversion
#    of the original (mlx_lm.convert, group size 64)
# 6. delete the downloaded weights and the hidden states (results stay)
#
# Strictly one model at a time; wrap a list of calls in one script and run
# power-guard.sh on 'verify-eu|measure-' for long queues.
# ─────────────────────────────────────────────────────────────────────────────
set -u
ID=$1; REF_REPO=$2; BUILD_REPO=$3; BITS=$4
HERE=${0:A:h}; PY=~/src/mlx/.venv/bin/python
M=~/src/mlx/models/eu; D=~/src/mlx/quality/$ID
REF=$M/${REF_REPO:gs/\//__}; BUILD=$M/${BUILD_REPO:gs/\//__}
Q=($PY $HERE/measure-quality.py --model generic --id $ID --ref $REF --build $BUILD)
log() { print -r -- "[$(date +%H:%M:%S)] $ID: $*"; }

export HF_HUB_DISABLE_XET=1
log "download"
$PY -c "
from huggingface_hub import snapshot_download as s
pat = ['*.json', '*.safetensors', '*.jinja', '*.model', '*.txt', 'tokenizer*', '*.tiktoken']
for r, d in (('$REF_REPO', '$REF'), ('$BUILD_REPO', '$BUILD')):
    s(r, local_dir=d, allow_patterns=pat)
" >/dev/null 2>&1 || { log "download failed"; exit 1; }

mkdir -p $D; [[ -d $D/raw ]] || cp -R ~/src/mlx/kolibri-quality/raw $D/
log "prepare"
$Q prepare --sets core,calib-v5,flores-eu,chat,tools 2>&1 | grep "set \|Error" | tail -12
log "check"
$Q check --ckpt ref 2>&1 | grep "max|dh|" | tee $D/check.txt
if grep -qv "max|dh| 0.000e+00" $D/check.txt; then log "WARNING: streamed forward not bit-identical, see $D/check.txt"; fi

for arm in ref ref-chunked build; do
  log "forward $arm"
  case $arm in
    ref) $Q forward --ckpt ref --arm ref ;;
    ref-chunked) $Q forward --ckpt ref --arm ref-chunked --chunk 512 ;;
    build) $Q forward --ckpt build --arm build ;;
  esac 2>&1 | grep -v "  layer\|Warning" | tail -2
done
log "report"
$Q report > $D/report.txt 2>&1; tail -1 $D/report.txt
R=$(ls -t $D/results/quality-*.json | head -1)
$PY $HERE/quality-summary.py $R --build $BUILD_REPO --arm build --ref $REF_REPO --noise ref-chunked \
  | $PY -c "import json,sys; d=json.load(sys.stdin); d['id']='$ID'; print(json.dumps(d))" >> ~/src/mlx/quality/summaries.jsonl
tail -1 ~/src/mlx/quality/summaries.jsonl | cut -c1-300

log "mmlu"
if [[ $BITS == 8 ]]; then E=$BUILD
else
  E=$M/$ID-8bit
  [[ -d $E ]] || $PY -m mlx_lm convert --hf-path $REF --mlx-path $E -q --q-bits 8 --q-group-size 64 >/dev/null 2>&1 \
    || { log "8-bit conversion failed"; E=""; }
fi
[[ -n $E ]] && $PY $HERE/measure-mmlu.py run --model $E --arm $ID-8bit --loader lm 2>&1 | tail -1

log "clean up"
rm -rf $REF $BUILD $M/$ID-8bit $D/hidden
log "done"
