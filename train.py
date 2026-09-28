#!/usr/bin/env python3
"""
Train a PPO agent to clear Super Mario Bros World 1 (levels 1-1 .. 1-4).

Phases:
  a     train on 1-1 only (learn basic running / jumping)
  b     fine-tune on all four world-1 levels, starting from phase A
  eval  evaluate a checkpoint: episodes per level, clear rate, mean x

Examples:
  ./venv/bin/python train.py --phase a
  ./venv/bin/python train.py --phase b --resume checkpoints/phase_a_final.zip
  ./venv/bin/python train.py --phase eval --resume checkpoints/phase_b_final.zip
  ./venv/bin/python train.py --phase eval --resume checkpoints/phase_b_final.zip --holdout
      (also evaluates on unseen holdout levels, e.g. 2-1, never trained on)
"""
import argparse
import glob
import os
import shutil
import sys

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from nes_py.wrappers import JoypadSpace

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")
LOG_DIR = os.path.join(HERE, "logs")
ROM_DIR = os.path.join(HERE, "roms")

LEVELS = [
    "SuperMarioBros-1-1-v0",
    "SuperMarioBros-1-2-v0",
    "SuperMarioBros-1-3-v0",
    "SuperMarioBros-1-4-v0",
]

# Levels the agent NEVER trains on: used only to test generalization
# of whatever it learned in World 1.
HOLDOUT_LEVELS = [
    "SuperMarioBros-2-1-v0",
]

# Agent acts every FRAMESKIP game frames.
FRAMESKIP = 4
# Hard cap on agent steps per episode (prevents infinite stalling).
MAX_STEPS = 9000
N_ENVS = 4


def ensure_rom():
    """Make sure a Super Mario Bros ROM is available.

    gym-super-mario-bros>=7 ships the vanilla ROM inside the package
    (``_roms/super-mario-bros.nes``). As a fallback, a user-supplied ``.nes``
    dump placed in ``roms/`` is copied into the package.
    """
    import gym_super_mario_bros

    pkg_dir = os.path.dirname(gym_super_mario_bros.__file__)
    bundled = os.path.join(pkg_dir, "_roms", "super-mario-bros.nes")
    if os.path.exists(bundled):
        return bundled

    candidates = sorted(glob.glob(os.path.join(ROM_DIR, "*.nes")))
    if not candidates:
        sys.exit(
            "No ROM found.\n"
            "This build of gym-super-mario-bros normally bundles the ROM, but it wasn't\n"
            "there. Drop your own Super Mario Bros .nes dump into:\n"
            f"  {ROM_DIR}/\n"
            "then re-run (any filename ending in .nes works)."
        )
    rom = candidates[0]
    os.makedirs(os.path.join(pkg_dir, "_roms"), exist_ok=True)
    shutil.copyfile(rom, bundled)
    print(f"Installed ROM -> {bundled}")
    return bundled


class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_r, done, info = 0.0, False, {}
        for _ in range(self._skip):
            obs, r, done, info = self.env.step(action)
            total_r += r
            if done:
                break
        return obs, total_r, done, info


class MarioShaping(gym.Wrapper):
    """Reward forward progress, big bonus for the flag, penalty for dying."""

    def __init__(self, env):
        super().__init__(env)
        self._best_x = 0
        self._steps = 0

    def reset(self, **kwargs):
        self._best_x = 0
        self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _game_reward, done, info = self.env.step(action)
        self._steps += 1
        x = info.get("x_pos", 0)

        reward = 0.0
        if x > self._best_x:  # only forward progress counts
            reward += 0.1 * (x - self._best_x)
            self._best_x = x
        if info.get("flag_get", False):
            reward += 50.0
        if done and not info.get("flag_get", False):
            reward -= 10.0  # died or ran out of time
        reward -= 0.002  # tiny time pressure: dawdling costs

        if self._steps >= MAX_STEPS:
            done = True
        return obs, reward, done, info


def make_env(level, rank=0, seed=0):
    def _init():
        env = gym_super_mario_bros.make(level)
        env = JoypadSpace(env, RIGHT_ONLY)
        env = FrameSkip(env, FRAMESKIP)
        env = MarioShaping(env)
        # NB: gym 0.21's ResizeObservation appends a channel dim to 2D input,
        # so resize while still RGB, then grayscale.
        env = ResizeObservation(env, 84)
        env = GrayScaleObservation(env, keep_dim=False)
        env = FrameStack(env, 4)
        env.seed(seed + rank)
        return env

    return _init


def make_vec(levels, seed=0):
    # observations are already channel-first (4, 84, 84) from FrameStack,
    # which is what SB3's CnnPolicy expects.
    venv = VecMonitor(SubprocVecEnv([make_env(lvl, rank=i, seed=seed) for i, lvl in enumerate(levels)]))
    return venv


class CycleLevelsEnv(gym.Wrapper):
    """Single env for EvalCallback that cycles through levels episode by episode."""

    def __init__(self, levels):
        self._levels = levels
        self._idx = -1
        super().__init__(make_env(levels[0])())

    def reset(self, **kwargs):
        self._idx = (self._idx + 1) % len(self._levels)
        self.env.close()
        self.env = make_env(self._levels[self._idx])()
        return self.env.reset(**kwargs)


def train_phase(levels, total_timesteps, name, resume=None, seed=0):
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # one env per level, cycling if fewer levels than envs
    env_levels = [levels[i % len(levels)] for i in range(N_ENVS)]
    vec_env = make_vec(env_levels, seed=seed)

    eval_env = VecMonitor(DummyVecEnv([lambda: CycleLevelsEnv(levels)]))

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=CKPT_DIR,
        log_path=LOG_DIR,
        eval_freq=max(50_000 // N_ENVS, 1000),
        n_eval_episodes=8,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=250_000 // N_ENVS,
        save_path=CKPT_DIR,
        name_prefix=f"{name}_ckpt",
    )

    if resume and os.path.exists(resume):
        print(f"Resuming from {resume}")
        model = PPO.load(resume, env=vec_env, tensorboard_log=LOG_DIR)
    else:
        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            n_steps=1024,
            batch_size=256,
            n_epochs=4,
            learning_rate=1e-4,
            gamma=0.9,
            gae_lambda=0.95,
            clip_range=0.1,
            vf_coef=0.5,
            ent_coef=0.01,
            max_grad_norm=0.5,
            seed=seed,
        )
    # reset_num_timesteps=False: keep the checkpoint's step counter so
    # restarts continue progress (6M total) instead of doing 6M more steps.
    model.learn(total_timesteps=total_timesteps, callback=[ckpt_cb, eval_cb],
                reset_num_timesteps=False)
    final = os.path.join(CKPT_DIR, f"{name}_final.zip")
    model.save(final)
    print(f"Saved {final}")
    vec_env.close()


def evaluate(checkpoint, episodes_per_level=5, include_holdout=False):
    from stable_baselines3.common.vec_env import DummyVecEnv

    levels = list(LEVELS)
    if include_holdout:
        levels += [lvl for lvl in HOLDOUT_LEVELS if lvl not in levels]

    results = {}
    for level in levels:
        venv = DummyVecEnv([make_env(level)])
        model = PPO.load(checkpoint, env=venv)
        clears, xs, scores = 0, [], []
        for _ in range(episodes_per_level):
            obs = venv.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _r, done, infos = venv.step(action)
            info = infos[0]
            clears += int(bool(info.get("flag_get", False)))
            xs.append(info.get("x_pos", 0))
            scores.append(info.get("score", 0))
        results[level] = dict(clears=clears, episodes=episodes_per_level,
                              mean_x=sum(xs) / len(xs), mean_score=sum(scores) / len(scores),
                              holdout=level in HOLDOUT_LEVELS)
        venv.close()
    print("\n=== Evaluation ===")
    for level, r in results.items():
        tag = "  [UNSEEN holdout]" if r["holdout"] else ""
        print(f"{level}: cleared {r['clears']}/{r['episodes']}  "
              f"mean_x={r['mean_x']:.0f}  mean_score={r['mean_score']:.0f}{tag}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["a", "b", "eval"], required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--holdout", action="store_true",
                    help="eval/record: also test on unseen holdout levels (never trained on)")
    args = ap.parse_args()

    ensure_rom()

    if args.phase == "a":
        train_phase([LEVELS[0]], args.timesteps or 6_000_000, "phase_a", args.resume)
    elif args.phase == "b":
        default_resume = os.path.join(CKPT_DIR, "phase_a_final.zip")
        train_phase(LEVELS, args.timesteps or 15_000_000, "phase_b",
                    args.resume or (default_resume if os.path.exists(default_resume) else None))
    elif args.phase == "eval":
        if not args.resume:
            sys.exit("--resume <checkpoint> is required for eval")
        evaluate(args.resume, args.episodes, include_holdout=args.holdout)


if __name__ == "__main__":
    main()
