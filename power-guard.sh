#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# Pause long measurement runs while the battery is low, resume once it has
# recharged.
#
# A small adapter cannot cover an M-series GPU under full load (35 W against
# well over 35 W in High Power Mode): the battery drains even on mains power,
# and on 2026-10-08 a run ended in hibernation at 1 %. This guard sends
# SIGSTOP to the matching processes when the charge falls below LOW (or the
# Mac runs on battery) and SIGCONT once it is back at HIGH on AC power. A
# stopped process uses no CPU or GPU, so the battery charges. If macOS holds
# the charge on AC ("not charging"), it resumes at LOW + 15 instead.
#
# Only for runs whose results do not depend on the clock (measure-quality.py).
# Never use it around timing measurements: a pause would end up in the numbers.
#
#   ./power-guard.sh                          # defaults below
#   LOW=25 HIGH=80 ./power-guard.sh 'measure-quality|measure-.*-quality'
#
# Stops by itself when no matching process has existed for a minute.
# ─────────────────────────────────────────────────────────────────────────────

set -u
PATTERN="${1:-measure-quality.py|measure-kolibri-quality.py|measure-apertus-quality.py}"
LOW="${LOW:-30}"
HIGH="${HIGH:-70}"
INTERVAL="${INTERVAL:-30}"

paused=0
idle=0
log() { print -r -- "[$(date +%H:%M:%S)] $*"; }

battery() {
  local out
  out=$(pmset -g batt)
  charge=$(print -r -- "$out" | grep -oE '[0-9]+%' | head -1 | tr -d '%')
  if print -r -- "$out" | grep -q "AC Power"; then on_ac=1; else on_ac=0; fi
  # macOS may hold the charge on AC ("not charging", battery health management);
  # waiting for HIGH would then wait forever.
  if print -r -- "$out" | grep -q "not charging"; then holding=1; else holding=0; fi
}

log "guarding '$PATTERN': pause below ${LOW}% or on battery, resume at ${HIGH}% on AC"
while true; do
  pids=($(pgrep -f -- "$PATTERN" | grep -v "^$$\$"))
  if (( ${#pids} == 0 )); then
    (( idle += INTERVAL ))
    (( idle >= 60 )) && { log "no matching process, exiting"; exit 0; }
    sleep "$INTERVAL"; continue
  fi
  idle=0
  battery
  if (( ! paused )) && { (( charge < LOW )) || (( ! on_ac )); }; then
    kill -STOP "${pids[@]}" 2>/dev/null && paused=1
    log "paused ${pids[*]} at ${charge}% ($([[ $on_ac == 1 ]] && echo AC || echo battery))"
  elif (( paused )) && (( on_ac )) && { (( charge >= HIGH )) || { (( holding )) && (( charge >= LOW + 15 )); }; }; then
    kill -CONT "${pids[@]}" 2>/dev/null && paused=0
    log "resumed ${pids[*]} at ${charge}%$( (( holding )) && echo ', macOS holds the charge')"
  elif (( paused )); then
    # Processes started while paused (the next step of a chain) wait too.
    kill -STOP "${pids[@]}" 2>/dev/null
  fi
  sleep "$INTERVAL"
done
