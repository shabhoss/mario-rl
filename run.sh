#!/usr/bin/env bash
# Run the full World-1 curriculum: phase A (1-1) then phase B (1-1..1-4).
# Safe to re-run after a crash: each phase resumes from its latest checkpoint.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python

latest_ckpt() { ls -t checkpoints/$1 2>/dev/null | head -1; }

# Resume phase A from whichever checkpoint is freshest: the periodic
# phase_a_ckpt (every 250k steps) or best_model.zip (saved on eval
# improvement, often newer). Prevents re-losing progress on restarts.
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

if [ $STATUS_A -eq 0 ]; then
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
else
  echo "phase A failed (exit $STATUS_A); not starting phase B" >> train.log
  exit $STATUS_A
fi
