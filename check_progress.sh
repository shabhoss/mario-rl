#!/usr/bin/env bash
# Periodic Mario RL training monitor. Reports status; relaunches if crashed.
cd ~/workspace/mario-rl || exit 1
echo "=== $(date '+%F %T') ==="
if pgrep -f "train.py --phase" >/dev/null; then
  echo "status: RUNNING"
else
  echo "status: NOT RUNNING"
  # relaunch if there's still work to do (phase B final not present)
  if [ ! -f checkpoints/phase_b_final.zip ]; then
    echo "relaunching run.sh"
    nohup ./run.sh >> train.log 2>&1 &
  else
    echo "phase_b_final.zip exists; work complete"
  fi
fi
# latest PPO timestep count from the log
ts=$(grep -oP 'total_timesteps\s+\|\s+\K[0-9]+' train.log 2>/dev/null | tail -1)
echo "total_timesteps: ${ts:-unknown}"
# latest eval best model
if [ -f checkpoints/best_model.zip ]; then
  echo "best_model: $(stat -c '%y %s bytes' checkpoints/best_model.zip)"
fi
ls checkpoints/*.zip 2>/dev/null | tail -4
# latest eval mean reward, if any
./venv/bin/python - "$PWD" << 'EOF' 2>/dev/null
import sys, os, glob
sys.path.insert(0, sys.argv[1])
logs = sorted(glob.glob(os.path.join(sys.argv[1], 'logs', 'evaluations.npz')))
if logs:
    import numpy as np
    d = np.load(logs[-1])
    r = d['results']
    print(f"eval_episodes: {len(r)}, last_mean_reward: {r[-1].mean():.1f}, best_mean: {r.mean(axis=1).max():.1f}")
EOF
