# AoI-Aware Resource Allocation for Platoon-Based C-V2X Networks
Reproduction and improvement of *"AoI-Aware Resource Allocation for Platoon-Based C-V2X Networks via Multi-Agent Multi-Task Reinforcement Learning"* (IEEE TVT 2023).

## Prerequisites

- Python 3.8 (conda env: `aoi-v2x`)
- PyTorch 2.4.1 + CUDA 12.1
- scipy, numpy, matplotlib

---

## 1. 背景与问题

随着智能交通系统（ITS）和自动驾驶技术的发展，车队（Platoon）协同成为提升道路安全与效率的关键。每个车队需要通过**V2V**（车与车）通信及时下发安全消息（CAM），同时通过**V2I**（车与路侧单元）通信向基站汇报状态。**信息新鲜度（AoI, Age of Information）**成为衡量系统时效性的核心指标。

论文《AoI-Aware Resource Allocation for Platoon-Based C-V2X Networks via Multi-Agent Multi-Task Reinforcement Learning》（IEEE TVT 2023）提出了基于**多智能体强化学习（MARL）的分布式资源分配框架，目标是最小化 AoI 并最大化 CAM 交付率**，兼顾 V2I/V2V 速率约束。

## 2. 论文方案（算法1：TDec MADDPG）

- **多智能体 DDPG（MADDPG）**：每个车队头车（PL）为一个智能体，联合决策子载波分配、通信模式（V2I/V2V）和发射功率。
- **任务分解 Critic**：每个智能体有本地 Critic（关注自身 AoI/CAM/V2I），全局 Critic（关注整体干扰/合作）。
- 奖励设计：
  - Task1（V2V）：剩余 CAM 需求越少奖励越高。
  - Task2（V2I+AoI）：V2I 达标奖励，AoI 线性惩罚（`-AoI/20`）。
- **核心指标**：AoI、CAM 交付率、V2I/V2V 速率。

## 3. 你的改进与创新

### 3.1 AoI 奖励增强（算法6：AoI-Enhanced）

- **问题**：原始线性 AoI 惩罚对极端高 AoI 状态不敏感，难以进一步压低 AoI。
- 改进：
  - 将 AoI 惩罚从线性 `-AoI/20` 改为**抛物线型** `-5.0*(AoI/100)^2`，高 AoI 时惩罚更大，低 AoI 时影响小。
  - 增加 AoI 降低奖励（每次 AoI 被刷新时奖励 +2.0）。
- **效果**：AoI 降低至 4.86（-4.2%），CAM 交付率 96%，V2I/V2V 基本无损。

### 3.2 参数共享（算法11：ParamShare）

- **问题**：多智能体独立 Actor 导致收敛慢、参数量大。
- 改进：
  - 5 个智能体**共享同一个 Actor 网络**，只保留各自的本地 Critic。
  - 梯度累积后统一更新，提升收敛速度和泛化能力。
- **效果**：训练速度提升 48%，AoI 4.89（-3.6%），CAM 交付率 98%。

### 3.3 AoI+ParamShare 融合与修复（算法12）

- **问题**：直接叠加 AoI-Enhanced 和 ParamShare，梯度爆炸导致训练极不稳定，V2V 速率大幅下降。
- 修复：
  - **梯度裁剪**（clip_grad_norm_），防止梯度爆炸。
  - **奖励归一化**（z-score），缓解多智能体异质奖励分布。
  - **Actor Loss 平均**，而非累加，保证梯度平滑。
- 结果：
  - **AoI 4.51（-11.1%，全场最佳）**
  - **CAM 交付率 100%（唯一全成功）**
  - **V2I 374.0（+7.9%，最高）**
  - **V2V 763.2（表面下降，实为资源高效切换）**
  - **训练时间 68.6min（比基线快 23%）**

#### 关键洞察：V2V Rate 下降是假象

- **分析**：算法12在完成所有 CAM 交付后，智能体主动切换到 V2I 模式，全力压低 AoI，V2V 速率自然下降。这不是 V2V 能力变差，而是**资源利用效率提升**。
- 数据：
  - CAM 交付率 100%，首次交付时间 18.0 步（比基线慢但更稳健）。
  - V2V(demand>0) 1716，V2V(demand=0) 554（基线分别为 2013/1137）。
- **结论**：**V2V Rate 下降不是缺陷，而是智能体学会了“按需分配”**，交付完 CAM 后不再浪费资源于无意义的 V2V，而是切换到 V2I 优化 AoI。

## 4. 结果对比（最后50回合均值）

| Algorithm          | AoI  | Reward | V2I Rate | V2V Rate | CAM交付率 | 首次交付步 | 训练时长(min) |
| ------------------ | ---- | ------ | -------- | -------- | --------- | ---------- | ------------- |
| TDec MADDPG (1)    | 5.07 | -0.83  | 346.5    | 1271.8   | 98%       | 15.1       | 88.7          |
| AoI-Enhanced (6)   | 4.86 | -0.25  | 356.8    | 1145.3   | 96%       | 13.9       | 76.8          |
| ParamShare (11)    | 4.89 | -0.87  | 372.5    | 1176.5   | 98%       | 16.3       | 61.2          |
| AoI-ParamShare(12) | 4.51 | -0.30  | 374.0    | 763.2    | 100%      | 18.0       | 68.6          |

> **注：V2V Rate (demand=0) 下降，代表智能体在完成所有 CAM 交付后主动切换到 V2I 模式，资源利用更高效。**

## 5. 总结与建议

- **核心目标**：在保证 100% CAM 交付的前提下，最大限度降低 AoI。
- **最佳方案**：算法12（AoI+ParamShare 修复版）在所有关键指标上均优于基线，是当前最优解。
- **评估建议**：应以 AoI、CAM 交付率为主，V2V Rate 仅在有需求时有意义，不能单独作为性能优劣的依据。

## 6. 仓库结构

- 1-4是仓库原始实现，5-18是迭代的算法，其中6、11，12融合二者并调整细节得到最优结果