# Mario RL — PPO agent for Super Mario Bros World 1

Trains a PPO agent (stable-baselines3) to clear levels 1-1 through 1-4 of the
original NES Super Mario Bros, using the `nes-py` emulator via
`gym-super-mario-bros`.

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install -r requirements.txt
```

**ROM:** the pinned `gym-super-mario-bros` release bundles the vanilla
Super Mario Bros ROM inside the package, so no extra step is needed. If you
ever swap to a build without it, drop your own `.nes` dump in `roms/` (any
filename) and the training script will install it where the package expects.

## Training

Two-phase curriculum:

- **Phase A** — 6M steps on 1-1 only. Learns to run right and jump.
- **Phase B** — 15M steps across 1-1..1-4, fine-tuned from Phase A.

```bash
./venv/bin/python train.py --phase a
./venv/bin/python train.py --phase b   # auto-resumes from phase_a_final.zip
```

Or everything in one go (logs to `train.log`):

```bash
./run.sh
```

Checkpoints land in `checkpoints/` (regular snapshots + best-eval model),
TensorBoard logs in `logs/`.

Reward shaping: +0.1 per pixel of new forward progress, +50 for grabbing the
flag, −10 for dying / timing out, tiny per-step time pressure. Action space is
`RIGHT_ONLY` (7 buttons) — enough to speedrun right. Observations are 84×84
grayscale ×4 frame stack.

## Evaluate / record

```bash
./venv/bin/python train.py --phase eval --resume checkpoints/phase_b_final.zip
./venv/bin/python record.py --checkpoint checkpoints/phase_b_final.zip --out videos/
```

`record.py` saves one mp4 per level showing the agent's playthrough.
