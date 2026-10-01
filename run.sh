#!/usr/bin/env bash
# Run the full World-1 curriculum: phase A (1-1) then phase B (1-1..1-4).
# Safe to re-run after a crash: each phase resumes from its latest checkpoint.
# Phase-aware: once phase_a_final.zip exists, phase A is never re-run (so a
# phase-B-era best_model.zip can never be mistaken for a phase-A checkpoint).
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python

latest_ckpt() { ls -t checkpoints/$1 2>/dev/null | head -1; }

if [ -f checkpoints/phase_b_final.zip ]; then
  echo "curriculum complete (checkpoints/phase_b_final.zip exists); nothing to do" >> train.log
  exit 0
fi

if [ ! -f checkpoints/phase_a_final.zip ]; then
  # Resume phase A from whichever phase-A checkpoint is freshest: the periodic
  # phase_a_ckpt (every 250k steps) or best_model.zip (saved on eval
  # improvement, often newer). Prevents re-losing progress on restarts.
  # (phase_a_final.zip missing means phase B never ran, so best_model.zip is
  # guaranteed to be phase-A-era here.)
  RESUME_A=""
  newest=""
  for c in checkpoints/phase_a_ckpt_*.zip checkpoints/best_model.zip; do
    [ -f "$c" ] || continue
    if [ -z "$newest" ] || [ "$c" -nt "$newest" ]; then newest="$c"; fi
  done
  [ -n "$newest" ] && RESUME_A="--resume $newest"

  # shellcheck disable=SC2086
  $PY train.py --phase a $RESUME_A >> train.log 2>&1
  STATUS_A=$?
  echo "phase A exit: $STATUS_A" >> train.log

  if [ $STATUS_A -ne 0 ]; then
    echo "phase A failed (exit $STATUS_A); not starting phase B" >> train.log
    exit $STATUS_A
  fi
else
  echo "phase A already complete; skipping to phase B" >> train.log
fi

RESUME_B=""
ckpt_b=$(latest_ckpt "phase_b_ckpt_*.zip")
if [ -n "$ckpt_b" ]; then
  RESUME_B="--resume $ckpt_b"
elif [ -f checkpoints/phase_a_final.zip ]; then
  RESUME_B="--resume checkpoints/phase_a_final.zip"
fi
# shellcheck disable=SC2086
$PY train.py --phase b $RESUME_B >> train.log 2>&1
STATUS_B=$?
echo "phase B exit: $STATUS_B" >> train.log
# Propagate a non-zero exit so process supervisors (systemd) restart on
# failure; exit 0 only when the full curriculum actually finished.
exit $STATUS_B
