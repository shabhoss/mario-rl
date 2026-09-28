#!/usr/bin/env python3
"""
Record the trained agent playing through World 1 as mp4 videos.

  ./venv/bin/python record.py --checkpoint checkpoints/phase_b_final.zip --out videos/
"""
import argparse
import os
import sys

import cv2
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import LEVELS, make_env, ensure_rom  # noqa: E402


def record(checkpoint, out_dir, fps=30):
    os.makedirs(out_dir, exist_ok=True)
    ensure_rom()
    for level in LEVELS:
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
            frames.append(base.render(mode="rgb_array"))
            cleared = bool(infos[0].get("flag_get", False))
            steps += 1

        path = os.path.join(out_dir, f"{level}.mp4")
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"{level}: {'CLEARED' if cleared else 'died'} in {steps} agent-steps -> {path}")
        venv.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos"))
    args = ap.parse_args()
    record(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
