#!/usr/bin/env python3
"""
Record the trained agent playing through World 1 as mp4 videos.

  ./venv/bin/python record.py --checkpoint checkpoints/phase_b_final.zip --out videos/
"""
import argparse
import os
import subprocess
import sys

import cv2
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import LEVELS, HOLDOUT_LEVELS, make_env, ensure_rom  # noqa: E402


def to_ios_friendly(src_path, dst_path):
    """Re-encode to H.264/yuv420p so the video plays on iOS."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", src_path,
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-crf", "20",
         dst_path],
        check=True,
    )
    os.remove(src_path)


def record(checkpoint, out_dir, include_holdout=False, levels=None, fps=30):
    os.makedirs(out_dir, exist_ok=True)
    ensure_rom()
    levels = list(levels) if levels else list(LEVELS)
    if include_holdout:
        levels += [lvl for lvl in HOLDOUT_LEVELS if lvl not in levels]
    for level in levels:
        venv = DummyVecEnv([make_env(level)])
        model = PPO.load(checkpoint, env=venv)
        raw_env = venv.envs[0].env  # innermost env for rendering
        # unwrap to the base gym env (below JoypadSpace) for render()
        base = raw_env
        while not isinstance(base, gym_super_mario_bros.SuperMarioBrosEnv):
            base = base.env

        frames = []
        obs = venv.reset()
        done, cleared, steps = False, False, 0
        while not done and steps < 20000:
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, done, infos = venv.step(action)
            # NB: render() returns a live view into the emulator's screen
            # buffer (same object every call), so copy it per frame.
            frames.append(base.render(mode="rgb_array").copy())
            cleared = bool(infos[0].get("flag_get", False))
            steps += 1

        raw_path = os.path.join(out_dir, f"{level}_raw.mp4")
        path = os.path.join(out_dir, f"{level}.mp4")
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        to_ios_friendly(raw_path, path)
        print(f"{level}: {'CLEARED' if cleared else 'died'} in {steps} agent-steps -> {path}")
        venv.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos"))
    ap.add_argument("--holdout", action="store_true",
                    help="also record the unseen holdout levels (never trained on)")
    ap.add_argument("--levels", default=None,
                    help="comma-separated level ids to record (default: all World 1 levels)")
    args = ap.parse_args()
    only = [l.strip() for l in args.levels.split(",")] if args.levels else None
    record(args.checkpoint, args.out, include_holdout=args.holdout, levels=only)


if __name__ == "__main__":
    main()
