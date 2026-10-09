#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# Aider Polyglot against the local mlx-vlm server (sovereign-models#14).
#
# 225 Exercism exercises in C++, Go, Java, JavaScript, Python and Rust; the
# model edits the code, the tests run, and a failed exercise gets a second try
# with the test output (pass@2). The official harness (Aider-AI/aider
# benchmark/) runs in its Docker image, because it executes model-written code;
# the model is served by start-mlx_kolibri.sh or start-mlx_apertus.sh on the
# host and reached as host.docker.internal.
#
#   ./aider-polyglot.sh kolibri dry-1 --num-tests 5     # dry run: time per exercise
#   ./aider-polyglot.sh kolibri run-1 --seed 1          # full run; repeat with seeds 2, 3
#   ./aider-polyglot.sh stats <run dir name>
#
# One-time setup (done 2026-10-09):
#   git clone https://github.com/Aider-AI/aider ~/src/aider-bench/aider
#   git clone https://github.com/Aider-AI/polyglot-benchmark \
#       ~/src/aider-bench/aider/tmp.benchmarks/polyglot-benchmark
#   (cd ~/src/aider-bench/aider && ./benchmark/docker_build.sh)
#
# SAMPLING per model card: Kolibri temperature 1.0, top_p 0.97, top_k 128 with
# reasoning_effort low (the client default of this repo); Apertus 1.5 gives no
# recommendation, so Apertus 1's temperature 0.8, top_p 0.9, thinking off (the
# template default). mlx-vlm seeds every request with the same default, so
# repeated runs need their own --seed or they are one sample three times.
#
# EDIT FORMAT "whole" (the model returns whole files): aider's recommendation
# for models it has no settings for; "diff" would be closer to the leaderboard
# entries of large models but fails more often for small ones. Override with
# EDIT_FORMAT=diff.
#
# Results: ~/src/aider-bench/aider/tmp.benchmarks/<date>--<name>/; the stats
# come from `benchmark.py --stats`. Server flags and versions go into
# <run dir>/server.txt.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
AIDER="${AIDER_DIR:-$HOME/src/aider-bench/aider}"
EDIT_FORMAT="${EDIT_FORMAT:-whole}"
THREADS="${THREADS:-1}"

if [[ "${1:-}" == "stats" ]]; then
  cd "$AIDER"
  exec docker run --rm -v "$PWD":/aider -v "$PWD/tmp.benchmarks/.":/benchmarks \
    -e AIDER_DOCKER=1 -e AIDER_BENCHMARK_DIR=/benchmarks aider-benchmark \
    ./benchmark/benchmark.py --stats "/benchmarks/$2"
fi

MODEL="${1:?kolibri or apertus}"
NAME="${2:?run name}"
shift 2
SEED=1
PASS=()
while (( $# )); do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    *) PASS+=("$1"); shift ;;
  esac
done

case "$MODEL" in
  kolibri)
    PORT=8888; SERVED="Kolibri-1-local"
    SAMPLING="use_temperature: 1.0
  extra_params:
    top_p: 0.97
    max_tokens: 16384
    extra_body: {top_k: 128, seed: $SEED, reasoning_effort: low}" ;;
  apertus)
    PORT=8890; SERVED="Apertus-v1.5-8B-local"
    SAMPLING="use_temperature: 0.8
  extra_params:
    top_p: 0.9
    max_tokens: 8192
    extra_body: {seed: $SEED}" ;;
  *) echo "unknown model $MODEL" >&2; exit 1 ;;
esac

curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || {
  echo "ERROR: no server on port $PORT; start start-mlx_$MODEL.sh first" >&2; exit 1; }

cd "$AIDER"
SETTINGS="tmp.benchmarks/model-settings-$MODEL-seed$SEED.yml"
cat > "$SETTINGS" <<YML
- name: openai/$SERVED
  edit_format: $EDIT_FORMAT
  $SAMPLING
YML

# What the server runs, for the record (the run dir is created by benchmark.py).
INFO="tmp.benchmarks/server-$MODEL-$NAME.txt"
{
  date '+%F %T'
  pmset -g batt | sed -n 1,2p
  pid=$(lsof -tiTCP:$PORT -sTCP:LISTEN | head -1)
  ps -o command= -p "$pid"
  py=$(ps -o command= -p "$pid" | awk '{print $1}')
  "$py" -c "import importlib.metadata as m;print('mlx', m.version('mlx'), 'mlx-vlm', m.version('mlx-vlm'))"
  echo "seed $SEED, edit format $EDIT_FORMAT, threads $THREADS"
  cat "$SETTINGS"
} > "$INFO" 2>&1

exec docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  -v "$PWD":/aider -v "$PWD/tmp.benchmarks/.":/benchmarks \
  -e OPENAI_API_BASE="http://host.docker.internal:$PORT/v1" -e OPENAI_API_KEY=local \
  -e AIDER_DOCKER=1 -e AIDER_BENCHMARK_DIR=/benchmarks \
  aider-benchmark \
  ./benchmark/benchmark.py "$NAME-$MODEL-seed$SEED" \
    --model "openai/$SERVED" --edit-format "$EDIT_FORMAT" \
    --read-model-settings "/aider/$SETTINGS" \
    --threads "$THREADS" --tries 2 --exercises-dir polyglot-benchmark "${PASS[@]}"
