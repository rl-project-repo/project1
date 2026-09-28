# CMPE 260 — Project 1: DQN on Atari Pong

**Team:** Akshata Madavi, Parth Maradia, Pratham Gala

Starting from the DQN baseline in *Deep Reinforcement Learning Hands-On* (Chapter 6),
this project reproduces the textbook's baseline, tries two alternatives to
epsilon-greedy exploration, and applies Prioritized Experience Replay (PER) to the
stronger of the two.

## Structure

```
lib/                          shared code: Atari preprocessing wrappers, DQN network
notebooks/
  dqn_pong_baseline.ipynb            standalone baseline notebook (Colab-ready)
  CMPE260_DQN_Pong_Baseline_Fixed.ipynb   earlier baseline-only notebook (superseded by the two below)
  CMPE260_DQN_Pong_Complete.ipynb    full pipeline: baseline + both alternatives + PER (in progress)
baseline_epsilon_seed42/      Step 1 results — epsilon-greedy DQN
boltzmann_softmax_seed42/     Step 2a results — annealed Boltzmann/softmax exploration
noisynet_seed42/              Step 2b results — NoisyNet exploration
smoke_seed42/                 quick pipeline sanity-check run (not a real training result)
```

Each experiment folder contains `results.json` (final metrics), `evaluation_*.json`
(30-episode greedy/near-greedy evaluation of the best checkpoint), and a
`tensorboard/` event file. Model checkpoints (`best.pt`, `latest.pt`, 7–66 MB each)
are **not** committed here — see [Checkpoints](#checkpoints) below.

## Setup

```bash
pip install gymnasium ale-py opencv-python-headless torch
```

Training is CPU-bound (single-threaded Atari emulation + preprocessing), so GPU
choice matters less than you'd expect — see Results for why an L4/T4 vs. a more
powerful GPU doesn't change wall-clock time much here.

## Results

| Run | Algorithm | GPU | Status | Frames | Episodes | Wall-clock | Mean reward (100-ep) |
|---|---|---|---|---|---|---|---|
| `baseline_epsilon_seed42` | epsilon-greedy | NVIDIA L4 | hit 2M-frame cap, never crossed 19.0 | 2,000,000 | 1,122 | 194.5 min | 18.97 (best 18.98) |
| `boltzmann_softmax_seed42` | Boltzmann/softmax, annealed temp 2.0→0.1 | Tesla T4 | solved | 1,035,135 | 631 | 120.4 min | 19.00 |
| `noisynet_seed42` | NoisyNet (factorized Gaussian noise) | Tesla T4 | solved | 447,036 | 237 | 68.5 min | 19.11 |

Textbook baseline claim for comparison: ~10 minutes to mean reward ~19 on a GTX 1080 Ti.

**Convergence-speed comparison** (frames to a common mean-reward-100 = +15 threshold,
since the baseline never formally solved within its frame budget):

| Method | Frames to +15 | vs. baseline | Normalized AUC |
|---|---|---|---|
| NoisyNet | 394,997 | **+47.9% faster** | 0.294 |
| Baseline | 757,740 | — | 0.048 |
| Boltzmann/softmax | 783,069 | −3.3% (slightly slower) | 0.031 |

**Takeaway:** NoisyNet is the empirically stronger Step-2 method — it converges
~48% faster than the baseline and reaches a higher final score. Boltzmann/softmax
did not show a convergence-speed improvement over the baseline despite matching it
on final score; this is called out honestly rather than assumed away, per the
assignment's own instruction not to claim a 10% gain unless the data supports it.

## Status

- [x] Step 1 — baseline (epsilon-greedy DQN)
- [x] Step 2 — two exploration alternatives (Boltzmann/softmax, NoisyNet), compared against baseline
- [ ] Step 3 — Prioritized Experience Replay on NoisyNet (the Step-2 winner). **Implemented and unit-validated, not yet trained to completion** — this is the one remaining experiment.

## Team contributions

| Member | Owns |
|---|---|
| Akshata Madavi | Baseline (Step 1) + shared infrastructure: preprocessing wrappers, DQN network, checkpoint/TensorBoard pipeline |
| Parth Maradia | Step 2a — Boltzmann/softmax exploration |
| Pratham Gala | Step 2b — NoisyNet exploration, and Step 3 — PER (built on NoisyNet) |

## Checkpoints

`.pt` checkpoint files are excluded from git (see `.gitignore`) — they're 7–66 MB
each and git isn't a good place for binary model weights. They're kept on the
team's shared Google Drive; ask a team member for access if you need to reload a
trained model rather than just the logged metrics.

## References

1. Mnih et al. (2015), *Human-level control through deep reinforcement learning*.
2. Schaul et al. (2016), *Prioritized Experience Replay*.
3. Fortunato et al. (2018), *Noisy Networks for Exploration*.
4. Lapan, *Deep Reinforcement Learning Hands-On*, Chapter 6 (Packt).
