# CLAUDE.md — AoI-V2X-IEEE-TVT-2023

## Project Overview

Reproduction and improvement of the paper "AoI-Aware Resource Allocation for Platoon-Based C-V2X Networks via Multi-Agent Multi-Task Reinforcement Learning" (IEEE TVT 2023).

The system models 5 vehicle platoons (20 vehicles, 4 per platoon) in an urban grid. Each platoon leader selects a resource block, communication mode (inter/intra-platoon), and transmit power to minimize Age of Information (AoI) while satisfying V2V/V2I rate requirements.

## Environment

- **Conda env**: `aoi-v2x` (Python 3.8)
- **Python path**: `/home/chenhuaji@corp.sse.tongji.edu.cn/.conda/envs/aoi-v2x/bin/python`
- **Machine**: 8x NVIDIA RTX 3090 (24GB each), 251GB RAM
- **Proxy**: Run `proxy` before pip install or git push
- **Training time**: ~60 min per algorithm (500 episodes, 100 steps each)

## Repository Layout

```
1-Modified MADDPG with TDec/     # Paper's proposed method (baseline, best performer)
2-Modified MADDPG/                # Without task decoupling
3-MADDPG_FDec/                    # Fully decentralized
4-DDPG/                           # Single-agent baseline
6-Modified MADDPG with AoI-Enhanced/  # Our improvement #1 (reward shaping, AoI=-4.2%)
9-Modified MADDPG with QMIX/         # QMIX value decomposition (failed)
10-Modified MADDPG with MASAC/        # Multi-agent SAC with entropy (failed)
11-Modified MADDPG with ParamShare/   # Our improvement #2 (param sharing, AoI=-3.6%)
12-Modified MADDPG with AoI-ParamShare/  # Combined AoI+ParamShare (failed, conflicting gradients)
13-Modified MADDPG with V2V-Bonus/       # V2V delivery bonus in task1 (in progress)
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

### Results (final 50-ep averages)
| Metric | TDec (Baseline) | AoI-Enhanced | Change |
|--------|:---------------:|:------------:|:------:|
| AoI | 5.07 | **4.86** | -4.2% |
| Task1+2 Reward | -0.83 | **-0.25** | +69.9% |
| V2I Rate | 346.5 | **356.8** | +3.0% |
| V2V Rate | 1271.8 | 1145.3 | -10.0% |

## Algorithm 11 — ParamShare (Our Improvement #2)

Based on `1-Modified MADDPG with TDec` with parameter sharing:

### `local_critic.py`
- All 5 agents share a single `ActorNetwork` instance (same parameters)
- Each agent still has its own local critics (task1, task2) — not shared
- `local_learn()` adds `accumulate_actor_grad` flag: only calls `backward()`, skips `optimizer.step()` and target update
- New `update_local_critic_targets()` and `update_shared_actor_target()` methods for separate updates

### `global_critic.py`
- Actor update: zero_grad ONCE → all agents `backward()` → step ONCE (gradients accumulated from all 5 agents)
- Target actor updated ONCE (not 5 times), preventing effective tau inflation

### `Main.py`
- Creates shared `ActorNetwork` + shared target `ActorNetwork` before agent loop
- Passes shared actors to all agents via constructor

### Results
| Metric | TDec (Baseline) | ParamShare | Change |
|--------|:---------------:|:----------:|:------:|
| AoI | 5.07 | **4.89** | -3.6% |
| Task1+2 Reward | -0.83 | -0.87 | -4.8% |
| V2I Rate | 346.5 | **372.5** | +7.5% |
| V2V Rate | 1271.8 | 1176.5 | -7.5% |

## Quantitative Comparison Table

| Algorithm | AoI | Task1+2 | V2I Rate | V2V Rate | Conv.Ep | Ep100 AoI | Time(min) |
|-----------|:---:|:-------:|:--------:|:--------:|:-------:|:---------:|:---------:|
| TDec MADDPG (1) | 5.07 | -0.83 | 346.5 | 1271.8 | 120 | 6.13 | 88.7 |
| MADDPG (2) | 7.10 | -1.34 | 389.2 | 549.5 | 355 | 42.67 | 73.2 |
| FDec MADDPG (3) | 8.86 | -1.35 | 263.3 | 803.9 | 386 | 61.69 | 79.8 |
| DDPG (4) | 42.84 | -4.07 | 289.8 | 1013.6 | 196 | 75.70 | 14.4 |
| **AoI-Enhanced (6)** | **4.86** | **-0.25** | **356.8** | 1145.3 | 340 | 10.46 | 76.8 |
| **ParamShare (11)** | **4.89** | -0.87 | **372.5** | 1176.5 | **81** | **5.36** | **61.2** |

Training time from `train.log` last line.

## Figures

- `plot_fig3_task_rewards.py` → `figures/fig3_task_rewards.png`: Task1/Task2 convergence per Platoon (TDec + AoI-Enhanced + ParamShare)
- `plot_fig4_reward_comparison.py` → `figures/fig4_reward_comparison.png`: Total reward convergence (algos 1-4 + 6, 11)
- `plot_fig5_aoi_convergence.py` → `figures/fig5_aoi_convergence.png`: AoI convergence (5 multi-agent algos + DDPG inset)
- `plot_fig8_platoon_behavior.py` → `figures/fig8_platoon_behavior.png`: Single platoon 100-step power + demand (TDec/AoI-Enhanced/ParamShare)
- `plot_table_comparison.py` → `figures/table_comparison.txt`: Quantitative table + LaTeX

## Reward Structure Analysis (TDec baseline)

- **Task 1 reward** (V2V demand):
  - `mode==0`: `(-4.95) * (Demand / V2V_demand_size)`
  - `mode==1`: `(-4.95) * (Demand / V2V_demand_size) - 0.5 * log(power, 5)`
  - Penalizes remaining V2V demand → indirectly encourages CAM delivery
- **Task 2 reward** (AoI + V2I):
  - `mode==0`: `0.05 * Revenue(V2I_rate, V2I_min) - 0.5 * log(power, 5) - AoI/20`
  - `mode==1`: `0.05 * Revenue(V2I_rate, V2I_min) - AoI/20`
  - Revenue = 1 if V2I >= threshold, else 0. AoI penalized linearly.
- **V2V_success** = `1 - active_links/n_platoon` — **computed but NOT in reward**
- **V2V/CAM gap**: Both improvements (6, 11) sacrifice V2V rate (~8-10% drop). No explicit CAM delivery bonus exists in the reward.

## Common Gotchas

- numpy deprecation: use `dtype=bool` (not `np.bool`), `dtype=np.int64` (not `np.int`)
- Training 3+ MADDPG variants in parallel may trigger OOM killer (each ~1.5GB RAM). Run sequentially if RAM < 50GB free.
- `.mat` results and model checkpoints are gitignored. Only source code and plots are tracked.
- Results saved as float16 in .mat files — precision loss in detailed analysis.

## Attempted Improvements — Experiment Log

### Succeeded

| # | Name | AoI | Task1+2 | V2I | V2V | Key Idea |
|---|------|:---:|:------:|:---:|:---:|----------|
| 6 | AoI-Enhanced v2 | **4.86** | **-0.25** | **356.8** | 1145.3 | Quadratic AoI penalty + reduction bonus + extended state |
| 11 | ParamShare | **4.89** | -0.87 | **372.5** | 1176.5 | Shared actor (parameter sharing) with gradient accumulation across all 5 agents |

### In Progress

| # | Name | Based on | Key Idea | Status |
|---|------|----------|----------|--------|
| 13 | V2V-Bonus | TDec (1) | Add V2V delivery completion bonus to task1 reward: `+3.0 * (remaining_time / time_slow)` when `Demand <= 0`. Directly incentivizes CAM delivery speed. | Training on GPU 0 |

### Failed — with reasons

| # | Name | AoI | Reward | What was tried | Why it failed |
|---|------|:---:|:------:|---------------|--------------|
| 5 | Attention | 5.64 | -0.90 | Multi-Head Self-Attention in Global Critic (replacing flat concatenation) | 5 agents too few — concatenation already captures all interactions. Attention only added optimization difficulty without expressiveness gain. |
| 7 | NoiseDecay | 5.44 | -0.74 | Cosine annealing of exploration noise (0.3→0.05) | Early convergence was good (ep100 AoI=4.6 vs baseline 6.4) but late-stage exploration collapsed — agents got stuck in local optima. Fixed schedule can't adapt to learning dynamics. |
| 5 | PER | 5.79 | -0.91 | Prioritized Experience Replay (TD-error based sampling + importance weights) | TD-error prioritization introduces bias in multi-agent setting. The importance weight correction was insufficient, causing unstable training. Uniform sampling works fine at 50k buffer / 5 agents. |
| 8 | Nstep-TD | 6.50 | -1.08 | 5-step TD return (accumulate 5 rewards + bootstrap at step 5) | Two issues: (1) 5-step span in 100-step episode is too aggressive, target variance is high; (2) rewards stored as float16, multi-step accumulation causes severe precision loss. |
| 9 | QMIX | 5.91 | 0.72 | Value decomposition: per-agent Q-networks + monotonic mixing network replacing global critic | Showed strong early convergence (ep200 AoI=4.56, best seen) but destabilized late (ep499 AoI=6.26). Monotonicity constraint (abs() weights) too restrictive — the real value function is non-monotonic (agent Q can decrease while total Q increases). V2V rate collapsed (929 vs 1272). |
| 10 | MASAC | 5.59 | 0.69 | Gaussian stochastic policy with entropy regularization (α=0.2) replacing fixed-noise DDPG | Squashed Gaussian policy fundamentally incompatible with this action space: RB/mode selection are effectively discrete decisions mapped through tanh, where stochastic exploration produces inconsistent discrete mappings. V2V rate catastrophically low (661 vs 1272) — stochastic actions prevent consistent V2V packet delivery. |
| 12 | AoI+ParamShare | 5.54 | 0.73 | Combined Algorithm 6 (AoI reward shaping + extended state) with Algorithm 11 (parameter sharing) | Training highly unstable: ep200 AoI=14.01 (diverged), recovered to 4.40 at ep499 but final 50-ep avg=5.54 (+9.3%). The AoI reward shaping changes the task2 signal distribution — with shared actor receiving averaged gradients from all agents, the heterogeneous task2 rewards (different AoI/peak per agent) create conflicting gradient directions. V2V collapsed to 838 (vs baseline 1272). Two improvements that work individually can conflict when combined. |
| 11-v1 | ParamShare (naive) | — | — | Shared actor with sequential per-agent zero_grad→backward→step | Each agent's optimizer.step() overwrites previous agent's update. Gradients from 5 agents conflict, task1 rewards consistently worse. Training unstable from the start. |

### Deleted experiments (not in repo)

| # | Name | Note |
|---|------|------|
| 5-v1 | Attention | Replaced by PER version |
| 6-v1 | AoI-Enhanced v1 | Peak AoI penalty was constant (always 0.3), exponential AoI penalty was weaker than linear. Fixed in v2. |
| 7 | Attention+AoI | Combined v1 bugs from both 5 and 6. Deleted. |

## Git

Remote: `https://github.com/Funny0101/AoI-V2X-IEEE-TVT-2023.git`
Branch: `main`
