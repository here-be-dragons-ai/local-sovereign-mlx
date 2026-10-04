#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# Interactive terminal chat with Kolibri 1 -- no server, the model is loaded
# into this process (32.8 GiB). Same checkpoint and prerequisites as
# start-mlx_kolibri.sh: patch 0050, iogpu.wired_limit_mb >= 40960, and no
# MLX server running at the same time (both do not fit into memory).
#
# Commands inside the chat:
#   /think none|low|medium|high   reasoning effort (default: none)
#   /show on|off                  print the thinking block (default: on)
#   /temp <t>                     sampling temperature (default 1.0, the
#                                 model card's recommendation; 0 = greedy)
#   /system <text>                set a system prompt (resets the chat)
#   /reset                        start a new conversation
#   /exit                         quit (Ctrl-D works too)
#   """                           start/end a multi-line message
#
# The KV cache is kept across turns, so a follow-up only prefills the new
# tokens. The reasoning effort is part of Kolibri's system prompt: changing
# it with /think makes the next turn prefill the whole conversation again.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

VENV_PY="${MLX_VENV_PY:-$HOME/src/mlx/.venv/bin/python}"
MODEL_DIR="${MODEL_DIR:-${MLX_MODELS:-$HOME/src/mlx/models}/Kolibri-1-MLX-3bit}"

[[ -x "$VENV_PY" ]] || { echo "ERROR: no venv python at $VENV_PY (install-prereqs.sh)" >&2; exit 1; }
[[ -f "$MODEL_DIR/config.json" ]] || { echo "ERROR: no checkpoint at $MODEL_DIR (install-prereqs.sh --model kolibri)" >&2; exit 1; }
"$VENV_PY" -c "import mlx_vlm.models.kolibri1" 2>/dev/null || {
  echo "ERROR: mlx_vlm has no kolibri1 model -- run patches/apply-patches.sh (0050)." >&2
  exit 1
}
_WIRED_MB=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
if (( _WIRED_MB < 40960 )); then
  echo "ERROR: iogpu.wired_limit_mb = ${_WIRED_MB} (< 40960); the weights alone are 32.8 GiB." >&2
  echo "       sudo sysctl -w iogpu.wired_limit_mb=40960" >&2
  exit 1
fi
if pgrep -f "mlx_vlm.server" >/dev/null; then
  echo "ERROR: an mlx_vlm server is running; stop it first (both do not fit into memory)." >&2
  exit 1
fi

_CHAT=$(cat <<'PY'
import os, sys, time, warnings
warnings.filterwarnings("ignore")
try:
    import readline  # noqa: F401  line editing and history for input()
except ImportError:
    pass

import mlx.core as mx
from mlx_vlm import load, stream_generate
from mlx_vlm.generate.common import PromptCacheState

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
EFFORTS = ("none", "low", "medium", "high")

model_dir = sys.argv[1]
print(f"{DIM}loading {model_dir} ...{RESET}", flush=True)
t0 = time.time()
model, processor = load(model_dir)
tokenizer = getattr(processor, "tokenizer", processor)
print(f"{DIM}loaded in {time.time() - t0:.1f}s, "
      f"{mx.get_active_memory() / 2**30:.1f} GiB active. /exit to quit.{RESET}\n")

state = {"effort": "none", "show": True, "temp": 1.0, "system": None}
messages, cache = [], PromptCacheState()


def reset():
    global messages, cache
    messages = [{"role": "system", "content": state["system"]}] if state["system"] else []
    cache = PromptCacheState()


def read_message():
    line = input(f"{BOLD}du ›{RESET} ")
    if line.strip() != '"""':
        return line
    lines = []
    while (line := input()).strip() != '"""':
        lines.append(line)
    return "\n".join(lines)


def command(text):
    cmd, _, arg = text[1:].partition(" ")
    arg = arg.strip()
    if cmd in ("exit", "quit"):
        raise EOFError
    if cmd == "reset":
        reset(); print(f"{DIM}new conversation{RESET}")
    elif cmd == "think" and arg in EFFORTS:
        state["effort"] = arg; print(f"{DIM}reasoning effort: {arg}{RESET}")
    elif cmd == "show" and arg in ("on", "off"):
        state["show"] = arg == "on"; print(f"{DIM}thinking shown: {arg}{RESET}")
    elif cmd == "temp":
        try:
            state["temp"] = float(arg); print(f"{DIM}temperature: {state['temp']}{RESET}")
        except ValueError:
            print(f"{DIM}/temp <number>{RESET}")
    elif cmd == "system":
        state["system"] = arg or None; reset()
        print(f"{DIM}system prompt {'set' if arg else 'cleared'}, new conversation{RESET}")
    else:
        print(f"{DIM}commands: /think {'|'.join(EFFORTS)} · /show on|off · /temp <t> · "
              f"/system <text> · /reset · /exit{RESET}")


def reply(user_text):
    messages.append({"role": "user", "content": user_text})
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        reasoning_effort=state["effort"],
    )
    out, last, thinking = "", None, False
    for chunk in stream_generate(
        model, processor, prompt,
        max_tokens=16384, temperature=state["temp"], top_p=0.97, top_k=128,
        prompt_cache_state=cache,
    ):
        last = chunk
        piece = chunk.text
        out += piece
        if "<think>" in piece:
            thinking = True
            piece = piece.replace("<think>", "")
            if state["show"]:
                sys.stdout.write(DIM)
        if "</think>" in piece:
            before, _, after = piece.partition("</think>")
            if state["show"]:
                sys.stdout.write(before + RESET + "\n")
            thinking, piece = False, after.lstrip("\n")
        if not thinking or state["show"]:
            sys.stdout.write(piece)
            sys.stdout.flush()
    sys.stdout.write(RESET + "\n")

    reasoning, _, answer = out.rpartition("</think>") if "</think>" in out else ("", "", out)
    msg = {"role": "assistant", "content": answer.strip()}
    if reasoning:
        msg["reasoning"] = reasoning.replace("<think>", "").strip()
    messages.append(msg)
    if last is not None:
        print(f"{DIM}[{last.prompt_tokens} prompt tok @ {last.prompt_tps:.0f} t/s · "
              f"{last.generation_tokens} tok @ {last.generation_tps:.1f} t/s · "
              f"peak {last.peak_memory:.1f} GB · think {state['effort']}]{RESET}\n")


reset()
while True:
    try:
        text = read_message()
    except (EOFError, KeyboardInterrupt):
        print(); break
    if not text.strip():
        continue
    try:
        if text.startswith("/"):
            command(text)
        else:
            reply(text)
    except EOFError:
        break
    except KeyboardInterrupt:
        # Interrupted mid-answer: drop the unanswered turn and its cache.
        print(f"{RESET}\n{DIM}interrupted{RESET}")
        if messages and messages[-1]["role"] == "user":
            messages.pop()
        cache = PromptCacheState()
PY
)

exec "$VENV_PY" -c "$_CHAT" "$MODEL_DIR"
