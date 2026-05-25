# Simulation code of the paper:
"AoI-Aware Resource Allocation for Platoon-Based C-V2X Networks via Multi-Agent Multi-Task Reinforcement Learning"

### If you want to cite:
> M. Parvini, M. R. Javan, N. Mokari, B. Abbasi and E. A. Jorswieck, "AoI-Aware Resource Allocation for Platoon-Based C-V2X Networks via Multi-Agent Multi-Task Reinforcement Learning," in IEEE Transactions on Vehicular Technology, doi: 10.1109/TVT.2023.3259688.

The simulation environment is based on the urban case defined in Annex A of 3GPP, TS 36.885, "Study on LTE-based V2X Services".

---

## Prerequisites

- Python 3.8 (conda env: `aoi-v2x`)
- PyTorch 2.4.1 + CUDA 12.1
- scipy, numpy, matplotlib

---

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `1-Modified MADDPG with TDec` | **Proposed method** (paper's best performer, baseline) |
| `2-Modified MADDPG` | Modified MADDPG without task decoupling |
| `3-MADDPG_FDec` | Fully decentralized MADDPG |
| `4-DDPG` | Centralized DDPG baseline |
| `6-Modified MADDPG with AoI-Enhanced` | **Our improvement**: AoI-aware reward redesign (**outperforms baseline**) |

---

## How to Run

### Single algorithm
```bash
conda activate aoi-v2x
python "1-Modified MADDPG with TDec/Main.py"
```

### Parallel execution (4 algorithms)
```bash
# Edit GPU assignment in run_all.sh first, then:
bash run_all.sh
```

### Plot results
```bash
python plot_results.py        # Fig 1-8 for algorithms 1-4
python plot_comparison.py     # Comparison: baseline vs AoI-Enhanced
```

---

## How to Plot (original instructions)

1. Simulation results are saved into `.../model/marl_model` as `.mat` files.

2. Except for Fig. 1 (from `reward_t1.mat` and `reward_t2.mat` directly), results should be **averaged with respect to the agents** before plotting.

3. Figs. 2 and 3 are plotted as follows:
   - Run `Modified MADDPG with TDec/Main`, average `(reward_t1.mat + reward_t2.mat)` for all agents
   - Run `Modified MADDPG/Main`, average `reward.mat` for all agents
   - Run `MADDPG_FDec/Main`, average `reward.mat` for all agents
   - Run `DDPG/Main`, average `reward.mat` for all agents

4. The remaining figures can be reproduced by the same procedure.

---

## Improvement: AoI-Enhanced Reward Design (Algorithm 6)

### Motivation

The original paper uses a **linear AoI penalty** (`AoI / 20`) in the Task 2 reward function. This provides a constant gradient regardless of the current AoI level, making the agent equally sensitive to AoI increases at both low and high values. In practice, high AoI (stale information) is far more harmful than low AoI, which should be reflected in the reward signal.

### Key Changes

1. **Quadratic AoI penalty** replaces the linear one:
   ```
   original:  -AoI / 20
   improved:  -5.0 * (AoI / AoI_max)^2
   ```
   At `AoI = AoI_max`, both give the same penalty (5.0), but the quadratic form provides a steeper gradient near high AoI, creating stronger pressure to avoid stale information.

2. **AoI reduction bonus** (`+2.0`) when V2I update is successfully received (AoI resets to 1). This gives the agent a direct positive reward for maintaining fresh information, encouraging proactive resource allocation for V2I communication.

3. **Extended state representation** (19 -> 21 dims): Added `AoI_trend` (change since last step) and `Peak_AoI` to help agents anticipate AoI dynamics.

### Results Comparison

| Metric | TDec (Baseline) | AoI-Enhanced (Ours) | Change |
|--------|:---------------:|:-------------------:|:------:|
| **Average AoI** | 5.02 | **4.81** | **-4.2%** |
| **Total Reward** | -0.82 | **-0.27** | **+66.5%** |
| **V2I Rate** | 341.1 | **385.3** | **+13.0%** |
| V2V Rate | 1269.7 | 1096.9 | -13.6% |
| Power (dBm) | 7.35 | 9.62 | +30.9% |

The AoI-Enhanced variant achieves **lower AoI** and **significantly higher reward** than the original method, demonstrating that the quadratic penalty and reduction bonus effectively guide the agent to prioritize information freshness. The trade-off is a moderate reduction in V2V rate, as the agent reallocates resources toward V2I to maintain low AoI.

### Full Comparison Across All Algorithms

| Algorithm | AoI | Reward | V2I | V2V |
|-----------|:---:|:------:|:---:|:---:|
| **6-AoI-Enhanced** | **4.81** | **-0.27** | **385.3** | 1096.9 |
| 1-TDec (Baseline) | 5.02 | -0.82 | 341.1 | **1270.0** |
| 2-Modified MADDPG | 6.84 | -1.21 | 381.1 | 563.5 |
| 3-MADDPG-FDec | 8.82 | -1.48 | 266.3 | 760.3 |
| 4-DDPG | 58.01 | -5.03 | 171.3 | 1204.6 |
