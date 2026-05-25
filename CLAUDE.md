# CLAUDE.md — AoI-V2X-IEEE-TVT-2023

## Project Overview

Reproduction and improvement of the paper "AoI-Aware Resource Allocation for Platoon-Based C-V2X Networks via Multi-Agent Multi-Task Reinforcement Learning" (IEEE TVT 2023).

The system models 5 vehicle platoons (20 vehicles, 4 per platoon) in an urban grid. Each platoon leader selects a resource block, communication mode (inter/intra-platoon), and transmit power to minimize Age of Information (AoI) while satisfying V2V/V2I rate requirements.

## Environment

- **Conda env**: `aoi-v2x` (Python 3.8)
- **Python path**: `/home/chenhuaji@corp.sse.tongji.edu.cn/.conda/envs/aoi-v2x/bin/python`
- **Machine**: 8x NVIDIA RTX 3090 (24GB each), 251GB RAM
- **Proxy**: Run `proxy` before pip install or git push
- **Training time**: ~17 min per algorithm (500 episodes, 100 steps each)

## Repository Layout

```
1-Modified MADDPG with TDec/     # Paper's proposed method (baseline, best performer)
2-Modified MADDPG/                # Without task decoupling
3-MADDPG_FDec/                    # Fully decentralized
4-DDPG/                           # Single-agent baseline
6-Modified MADDPG with AoI-Enhanced/  # Our improvement (outperforms baseline)
figures/                          # Generated comparison plots
plot_results.py                   # Plot fig1-8 for algorithms 1-4
plot_comparison.py                # Plot baseline vs AoI-Enhanced
run_all.sh                        # Parallel training for 1-4
```

Each algorithm directory has the same structure:
- `Main.py` — Training entry point (hyperparams, training loop, state definition)
- `global_critic.py` — Global critic with twin Q-networks
- `local_critic.py` — Per-agent local critics (task1, task2) + actor
- `Classes/Environment_Platoon.py` — V2X environment (channels, rewards, AoI)
- `Classes/networks.py` — Actor and local critic networks
- `Classes/G_network.py` — Global critic network (only algos 1, 2, 6)
- `Classes/buffer.py` — Replay buffer
- `model/marl_model/` — Saved `.mat` results (gitignored)
- `Classes/tmp/ddpg/` — Saved model checkpoints (gitignored)

## Key Architecture Details

- **State dim**: 19 per agent (algo 1-5), 21 per agent (algo 6 adds AoI_trend + Peak_AoI)
- **Action dim**: 3 continuous (RB selection, mode selection, power) mapped via tanh -> discrete
- **Actor**: [input -> 1024 -> 512 -> 3] with LayerNorm + ReLU, tanh output
- **Local Critics**: [input -> 512 -> 256 -> 1] each for task1 and task2
- **Global Critic**: [concat_all_states(95) -> 1024 -> 512 -> 256 -> 1] with action branch
- **Hyperparams**: batch=64, memory=50k, gamma=0.99, tau=0.005, lr_actor=1e-4, lr_critic=1e-3, noise=0.3

## Algorithm 6 — AoI-Enhanced (Our Improvement)

Based on `1-Modified MADDPG with TDec` with changes in:

### `Classes/Environment_Platoon.py`
- Added `AoI_prev` and `Peak_AoI` tracking fields
- `Age_of_Information()`: now tracks previous AoI and peak AoI per platoon
- Reward function (`act_for_training`):
  - Replaced linear penalty `-AoI/20` with quadratic `-5.0*(AoI/100)^2`
  - Added AoI reduction bonus `+2.0` when V2I update received (AoI resets)
  - Removed failed Peak AoI penalty (was constant, provided no gradient)

### `Main.py`
- `get_state()`: added 2 features: `AoI_trend` (delta AoI normalized), `Peak_AoI` (normalized) -> state 19->21 dims
- Training loop: resets `Peak_AoI` to current `AoI` each episode (not fixed 100)
- Saves additional `Peak_AoI.mat` result file

### Results
| Metric | TDec (Baseline) | AoI-Enhanced | Change |
|--------|:---------------:|:------------:|:------:|
| AoI | 5.02 | **4.81** | -4.2% |
| Reward | -0.82 | **-0.27** | +66.5% |
| V2I Rate | 341.1 | **385.3** | +13.0% |
| V2V Rate | 1269.7 | 1096.9 | -13.6% |

## Common Gotchas

- numpy deprecation: use `dtype=bool` (not `np.bool`), `dtype=np.int64` (not `np.int`)
- Training 3+ MADDPG variants in parallel may trigger OOM killer (each ~1.5GB RAM). Run sequentially if RAM < 50GB free.
- `.mat` results and model checkpoints are gitignored. Only source code and plots are tracked.
- Results saved as float16 in .mat files — precision loss in detailed analysis.

## Git

Remote: `https://github.com/Funny0101/AoI-V2X-IEEE-TVT-2023.git`
Branch: `main`
