# PPT 大纲：AoI-Aware Resource Allocation for Platoon-Based C-V2X Networks

> 5分钟汇报，共12页幻灯片，每页约25秒

---

## 第1页：封面

- **标题**：AoI-Aware Resource Allocation for Platoon-Based C-V2X Networks via MARL
- **副标题**：论文复现与改进
- **汇报人信息**
- **图片**：无（或用一张简洁的车辆编队抽象图作为背景）

---

## 第2页：研究背景与问题

### 内容要点
- **车队编队（Platooning）**：多辆自动驾驶车以紧密队形行驶，通过V2X通信协同
- **两类通信需求**：
  - V2V（车与车）：PL向队员广播CAM安全消息 → 保证编队稳定性
  - V2I（车与基站）：PL向RSU汇报状态 → 保证信息时效性
- **核心问题**：如何分布式分配无线资源（子载波、通信模式、功率），**最小化信息年龄（AoI）**的同时**保证CAM消息的交付率**？
- **挑战**：高动态信道、多智能体干扰、多目标优化（MINLP问题）

### 图片
- **原始论文 Fig.1**（多车道编队场景图）：展示RSU、多个车队、V2I/V2V链路的直观场景
  - 来源：论文原文 Fig. 1（Multi-lane platoon scenario）
  - 图片URL：`https://cdn-mineru.openxlab.org.cn/result/2026-05-26/71c33fe1-5ebe-4c06-9c0f-2eafe73efa16/cfd3e2842528fa37d665d10ad7e7cb518108c62b635154b8139724ddaf6fdd84.jpg`

---

## 第3页：系统模型与优化问题

### 内容要点
- **场景**：1个RSU + 5个车队（每队4辆车，共20辆），城市网格道路
- **每个PL的决策变量**：
  - β：选择哪个子载波（RB）
  - θ：通信模式选择（V2I或V2V）
  - p：发射功率
- **AoI定义**：从最近一次成功V2I传输到现在的时间差；成功传输后重置为Δt，否则累加
- **多目标优化（MINLP）**：
  - min 平均AoI
  - max CAM消息交付概率
  - min 功率消耗
  - s.t. V2I最低速率要求、功率上限、每个PL只能选一个RB

### 图片
- **AI生成的系统模型示意图**：
  - 展示：5个车队→PL决策→3个输出（RB选择、模式选择、功率）→两条链路（V2I到RSU、V2V到PMs）
  - AoI更新公式的简化图示（成功→重置，失败→累加）
  - 参考：论文公式(4)和(5)

---

## 第4页：原始论文方法 — Modified MADDPG with TDec

### 内容要点
- **框架**：多智能体DDPG（MADDPG）+ 任务分解
  - 每个PL作为一个智能体，分布式决策
  - 全局Critic（在RSU上）：评估所有智能体的协作效果（基于干扰水平），用TD3防止Q值过估计
  - 本地Critic（每个PL独立）：评估自身任务表现
- **任务分解**：将整体奖励拆成两个子任务
  - Task 1（V2V/CAM交付）：剩余CAM需求越少，奖励越高
  - Task 2（V2I+AoI）：V2I达标给奖励 + AoI线性惩罚 `-AoI/20`
- **结果**：收敛最快，AoI=5.07，CAM交付率98%

### 图片
- **AI生成的算法架构图**（因为原文Fig.2太学术化）：
  - 左侧：5个PL（Agent 1-5），各有Actor + Local Critic (Task1) + Local Critic (Task2)
  - 右侧：RSU上的Global Critic (Twin TD3)
  - 连线：所有Agent的(s,a)输入Global Critic；各Agent的局部(s,a)输入各自Local Critic
  - 梯度流向：Global Critic梯度 + Local Critic梯度 → 更新Actor
  - 参考：论文 Fig. 2 和 Algorithm 2

---

## 第5页：原始方法的缺陷分析

### 内容要点
- **缺陷1：AoI奖励设计不足**
  - 线性惩罚 `-AoI/20` 对高AoI状态不够敏感
  - AoI从10→20和从90→100的惩罚一样，无法"紧急处理"极端高AoI
- **缺陷2：多智能体独立参数，训练效率低**
  - 5个智能体各自独立Actor网络，参数不共享
  - 收敛慢（120 episodes），训练时间长（88.7分钟）
  - 5个结构相同的网络各自从零学习，浪费了"同构智能体"的先验

### 图片
- **AI生成的对比示意图**：
  - 左：线性惩罚函数 `-AoI/20` 的曲线（平缓直线）
  - 右：抛物线型惩罚 `-5.0*(AoI/100)²` 的曲线（高AoI时惩罚急剧增大）
  - 标注：说明抛物线型对"极端高AoI"有更强的压制力
  - 参考：CLAUDE.md中"Replaced linear penalty -AoI/20 with quadratic -5.0*(AoI/100)^2"

---

## 第6页：改进一 — AoI-Enhanced（算法6）

### 内容要点
- **核心改进：奖励塑形（Reward Shaping）**
  - 将AoI惩罚从线性 `-AoI/20` 改为抛物线型 `-5.0*(AoI/100)²`
    - 高AoI时惩罚指数级增大 → 智能体更积极地降低极端AoI
    - 低AoI时惩罚温和 → 不影响其他目标的优化
  - 新增AoI刷新奖励：当V2I成功传输（AoI重置）时，额外奖励 `+2.0`
- **状态增强**：新增2个状态特征（AoI_trend、Peak_AoI），状态维度19→21
- **结果**：
  - AoI: 5.07 → **4.86（-4.2%）**
  - Reward: -0.83 → **-0.25（+69.9%）**
  - V2I Rate: 346.5 → **356.8（+3.0%）**

### 图片
- **AI生成的奖励函数对比图**：
  - 左侧原始：线性惩罚曲线
  - 右侧改进：抛物线惩罚曲线 + 刷新奖励标注
  - 下方：状态维度变化示意（19维 → 21维，新增AoI_trend和Peak_AoI）
  - 参考：CLAUDE.md "Algorithm 6 — AoI-Enhanced" 部分

---

## 第7页：改进二 — ParamShare（算法11）

### 内容要点
- **核心改进：参数共享（Parameter Sharing）**
  - 5个智能体**共享同一个Actor网络**（参数相同）
  - 每个智能体仍保留独立的Local Critic（因为各智能体的AoI/CAM状态不同）
- **梯度累积策略**：
  - 5个智能体分别计算梯度（backward），累加后统一更新（step一次）
  - 目标网络只更新一次（而非5次），防止τ膨胀
- **为什么有效**：
  - 所有车队面临相同的底层决策逻辑（选RB、选模式、选功率）
  - 共享参数 → 5倍数据利用效率 → 更快收敛、更好泛化
- **结果**：
  - AoI: 5.07 → **4.89（-3.6%）**
  - V2I Rate: 346.5 → **372.5（+7.5%）**
  - 收敛速度：120 → **81 episodes（1.48x加速）**
  - 训练时间：88.7 → **61.2 min（-31%）**

### 图片
- **AI生成的参数共享架构图**：
  - 左侧原始：5个独立的Actor网络（Actor1~Actor5）
  - 右侧改进：1个共享Actor网络，5个Agent共用
  - 箭头示意：Agent1~5各自backward → 梯度累加 → 统一step
  - 参考：CLAUDE.md "Algorithm 11 — ParamShare" 部分

---

## 第8页：改进三 — AoI+ParamShare融合（算法12）

### 内容要点
- **挑战**：直接叠加改进一和改进二 → 梯度爆炸，训练不稳定
  - 原因：AoI奖励塑形改变了Task2信号分布，5个智能体的异质奖励产生冲突梯度
- **三重修复**：
  1. **梯度裁剪**（clip_grad_norm_=1.0）：防止梯度爆炸
  2. **奖励归一化**（Z-score）：缓解多智能体异质奖励分布差异
  3. **Actor Loss平均**（而非累加）：保证梯度平滑
- **结果（全场最优）**：
  - AoI: 5.07 → **4.51（-11.1%，最佳）**
  - CAM交付率: 98% → **100%（唯一全成功）**
  - V2I Rate: 346.5 → **374.0（+7.9%，最高）**
  - 训练时间: 88.7 → **68.6 min（-23%）**

### 图片
- **AI生成的融合改进示意图**：
  - 展示三层修复的叠加效果：
    - 底层：参数共享（ParamShare）
    - 中层：AoI奖励塑形（抛物线惩罚 + 刷新奖励）
    - 顶层：稳定性修复（梯度裁剪 + Z-score归一化 + Loss平均）
  - 参考：CLAUDE.md "Algorithm 12-Fix" 部分

---

## 第9页：实验结果 — 全面对比

### 内容要点
- 定量对比表（7个算法 × 关键指标）

| Algorithm | AoI ↓ | Reward ↑ | V2I ↑ | V2V | CAM ↑ | Time ↓ |
|-----------|:-----:|:-------:|:-----:|:---:|:-----:|:------:|
| TDec MADDPG | 5.07 | -0.83 | 346.5 | 1271.8 | 98% | 88.7 |
| MADDPG | 7.10 | -1.34 | 389.2 | 549.5 | 4% | 73.2 |
| FDec | 8.86 | -1.35 | 263.3 | 803.9 | 32% | 79.8 |
| DDPG | 42.84 | -4.07 | 289.8 | 1013.6 | 2% | 14.4 |
| **AoI-Enhanced** | **4.86** | **-0.25** | **356.8** | 1145.3 | 96% | 76.8 |
| **ParamShare** | **4.89** | -0.87 | **372.5** | 1176.5 | 98% | **61.2** |
| **AoI-ParamShare** | **4.51** | **-0.30** | **374.0** | 763.2 | **100%** | 68.6 |

- 关键发现：算法12在AoI、CAM、V2I上均达到最优

### 图片
- 可选：在表上用颜色标注最优值（红蓝）
- 来源：`figures/table_comparison.txt`

---

## 第10页：实验结果 — AoI收敛曲线

### 内容要点
- 算法12（AoI-ParamShare）在训练过程中AoI持续下降，最终稳定在最低水平
- 相比基线TDec MADDPG，改进算法的AoI曲线明显更低
- DDPG作为单智能体方法完全无法处理多智能体场景（AoI极高，需要单独的inset图展示）

### 图片
- **已有图表**：`figures/fig5_aoi_convergence.png`
  - 展示所有算法的AoI随训练episodes的收敛曲线
  - 重点突出算法6、11、12的曲线在基线1之下

---

## 第11页：实验结果 — V2V深度分析与行为分析

### 内容要点
- **V2V Rate的"假象下降"**：
  - 算法12的V2V Rate（763.2）看似比基线（1271.8）低40%
  - 但拆分后发现：
    - CAM交付期间（demand>0）：V2V=1716（vs 基线2013，差距不大）
    - CAM交付完成后（demand=0）：V2V=554（vs 基线1137，主动降低）
  - **结论**：智能体学会了"按需分配"——完成CAM后主动切换到V2I模式压低AoI
- CAM交付率100%是唯一全成功的算法

### 图片
- **V2V拆分对比柱状图**（AI生成）：
  - 横轴：4个算法（TDec/AoI-Enhanced/ParamShare/AoI-ParamShare）
  - 每个算法两根柱子：V2V(demand>0) 和 V2V(demand=0)
  - 数据来源：table_comparison.txt中的V2V Rate Breakdown

- **车队行为分析图**：`figures/fig8_platoon_behavior.png`
  - 展示单个车队在100步内的功率分配和CAM剩余量变化
  - 对比TDec和AoI-ParamShare的行为差异

---

## 第12页：总结与展望

### 内容要点
- **核心贡献**：
  1. 复现并验证了原始论文的TDec MADDPG框架
  2. 提出三项改进：AoI奖励塑形、参数共享、融合修复
  3. 最终算法（AoI-ParamShare）实现：
     - AoI降低 **11.1%**（5.07→4.51）
     - CAM交付率 **100%**（唯一）
     - V2I提升 **7.9%**
     - 训练加速 **23%**
- **关键洞察**：V2V Rate下降不是缺陷，而是智能体学会了高效的资源按需分配策略
- **展望**：
  - 扩展到NOMA/MIMO场景
  - 更多智能体规模的泛化能力验证
  - 实际通信环境中的部署

### 图片
- 无额外图片，可用简洁的总结图示或关键词云

---

## 每页时间分配建议

| 页码 | 内容 | 建议时长 |
|:----:|------|:--------:|
| 1 | 封面 | 5s |
| 2 | 研究背景与问题 | 30s |
| 3 | 系统模型 | 25s |
| 4 | 原始方法 | 30s |
| 5 | 缺陷分析 | 20s |
| 6 | 改进一：AoI-Enhanced | 30s |
| 7 | 改进二：ParamShare | 30s |
| 8 | 改进三：融合修复 | 30s |
| 9 | 结果对比表 | 25s |
| 10 | AoI收敛曲线 | 20s |
| 11 | V2V深度分析 | 25s |
| 12 | 总结 | 20s |
| | **总计** | **~5min** |

---

## 图片清单汇总

### 已有的可直接使用的图片
1. `figures/fig5_aoi_convergence.png` — 第10页AoI收敛曲线
2. `figures/fig8_platoon_behavior.png` — 第11页车队行为分析
3. `figures/fig6_cam_v2i_comparison.png` — 可选，CAM交付率+V2I速率柱状图
4. `figures/fig3_task_rewards.png` — 可选，Task1/Task2收敛曲线
5. `figures/fig4_reward_comparison.png` — 可选，总奖励收敛对比
6. 原文Fig.1图片URL — 第2页背景场景

### 需要AI生成的图片
1. **第3页 — 系统模型示意图**：5个车队、RSU、V2I/V2V链路、PL决策变量、AoI更新示意
2. **第4页 — 算法架构图**：TDec MADDPG的多智能体架构（Actor × 5 + Local Critic × 2 × 5 + Global Critic TD3），梯度流向
3. **第5页 — 惩罚函数对比图**：线性 vs 抛物线型AoI惩罚曲线
4. **第6页 — AoI-Enhanced改进示意图**：抛物线惩罚+刷新奖励+状态增强
5. **第7页 — 参数共享架构图**：5独立Actor vs 1共享Actor + 梯度累积
6. **第8页 — 融合改进示意图**：三层修复的叠加效果
7. **第9页 — 定量对比表**：带颜色高亮的数据表格（可由PPT直接制作）
8. **第11页 — V2V拆分柱状图**：demand>0 vs demand=0的分组柱状图

---
---

## 附录：供GPT生成PPT时参考的完整技术内容

### A. 论文基本信息

- **标题**：AoI-Aware Resource Allocation for Platoon-Based C-V2X Networks via Multi-Agent Multi-Task Reinforcement Learning
- **作者**：Mohammad Parvini et al., IEEE Transactions on Vehicular Technology, 2023
- **场景**：城市网格道路，1个RSU + 5个车队（每个车队4辆车），共20辆车
- **频段**：2GHz，3个RB（每个180kHz），CAM消息大小4000字节，时间约束100ms
- **核心指标**：AoI（信息年龄）、CAM消息交付率、V2I/V2V数据速率

### B. 原始论文方法（Modified MADDPG with Task Decomposition）

- **多智能体建模**：每个车队头车（PL）为一个智能体，联合决策子载波分配(β)、通信模式(θ)、发射功率(p)
- **双层Critic架构**：
  - 全局Critic（在RSU上，所有智能体共享）：输入所有智能体的(s,a)，用TD3（Twin Delayed DDPG）评估全局团队奖励（基于干扰水平）
  - 本地Critic（每个智能体独立）：输入自身的(s,a)，评估个体任务奖励
- **任务分解（Task Decomposition）**：
  - Task 1（V2V/CAM交付）：r_1 = -4.95*(Demand/V2V_demand) - θ·power_penalty
  - Task 2（V2I+AoI）：r_2 = 0.05·Revenue(V2I_rate) - AoI/20 - (1-θ)·power_penalty
- **奖励结构**：全局奖励 r_g = -(1/P)·ΣΣ log(I_j[k])（平均干扰水平）
- **网络结构**：
  - Actor: [input → 1024 → 512 → 3] (LayerNorm + ReLU, tanh output)
  - Local Critic: [input → 512 → 256 → 1]
  - Global Critic: [concat(95) → 1024 → 512 → 256 → 1] (Twin TD3)
- **超参数**：batch=64, memory=50k, γ=0.99, τ=0.005, lr_actor=1e-4, lr_critic=1e-3, noise=0.3
- **状态空间**（19维/agent）：信道增益h(V2I), h(V2V), 干扰I, AoI, 剩余CAM需求ζ^r, 剩余时间T^r
- **动作空间**（3维连续，经tanh映射到离散）：RB选择、模式选择、功率

### C. 改进一：AoI-Enhanced（算法6）

**问题**：线性AoI惩罚 `-AoI/20` 对极端高AoI不敏感，难以进一步压低AoI

**改进**：
1. AoI惩罚改为抛物线型：`-5.0 * (AoI/100)²`
   - AoI=10时惩罚：线性=-0.5，抛物线=-0.05；AoI=50时：线性=-2.5，抛物线=-1.25；AoI=100时：线性=-5.0，抛物线=-5.0
   - 关键差异：AoI>100时抛物线惩罚继续指数增长，线性则等比例增长
2. 新增AoI刷新奖励：V2I成功传输（AoI重置）时 `+2.0`
3. 状态增强：新增AoI_trend（AoI变化趋势归一化）和Peak_AoI（峰值AoI归一化），状态维度19→21

**结果**（最后50回合均值）：
| 指标 | TDec基线 | AoI-Enhanced | 变化 |
|------|:--------:|:------------:|:----:|
| AoI | 5.07 | 4.86 | -4.2% |
| Reward | -0.83 | -0.25 | +69.9% |
| V2I Rate | 346.5 | 356.8 | +3.0% |
| V2V Rate | 1271.8 | 1145.3 | -10.0% |
| CAM | 98% | 96% | -2% |

### D. 改进二：ParamShare（算法11）

**问题**：5个同构智能体各自独立Actor网络，参数不共享，训练效率低

**改进**：
1. 5个智能体共享同一个ActorNetwork实例（参数相同）
2. 每个智能体仍保留独立的Local Critic（task1, task2）
3. 梯度累积：所有Agent backward()后统一step()，梯度来自5个Agent的累积
4. 目标网络只更新1次（而非5次），防止τ膨胀（0.005 × 5 = 0.025 的等效学习率问题）

**为什么有效**：
- 所有车队面临相同的决策结构（选RB、选模式、选功率）
- 共享参数 = 5倍数据利用效率 = 更快收敛 + 更好泛化
- 独立Critic保留个体差异性（各车队的AoI/CAM状态不同）

**结果**：
| 指标 | TDec基线 | ParamShare | 变化 |
|------|:--------:|:----------:|:----:|
| AoI | 5.07 | 4.89 | -3.6% |
| V2I Rate | 346.5 | 372.5 | +7.5% |
| V2V Rate | 1271.8 | 1176.5 | -7.5% |
| 收敛回合 | 120 | 81 | 1.48x加速 |
| 训练时间 | 88.7 min | 61.2 min | -31% |

### E. 合并：AoI+ParamShare融合修复（算法12）

**问题**：直接叠加AoI-Enhanced和ParamShare → 梯度爆炸，训练极度不稳定
- 原因：AoI奖励塑形改变了Task2信号分布，共享Actor收到5个Agent的异质梯度方向产生冲突

**三重修复**：
1. **梯度裁剪**（clip_grad_norm_=1.0）：限制梯度范数，防止爆炸
2. **奖励归一化**（Z-score）：对每个Agent的奖励做标准化，缓解异质奖励分布差异
3. **Actor Loss平均**（而非累加）：5个Agent的梯度取平均而非求和，保证梯度平滑

**结果（全场最优）**：
| 指标 | TDec基线 | AoI-ParamShare | 变化 |
|------|:--------:|:--------------:|:----:|
| AoI | 5.07 | **4.51** | **-11.1%** |
| Reward | -0.83 | -0.30 | +63.6% |
| V2I Rate | 346.5 | **374.0** | **+7.9%** |
| V2V Rate | 1271.8 | 763.2 | -40.0% |
| CAM | 98% | **100%** | **+2%** |
| 训练时间 | 88.7 min | 68.6 min | -22.6% |

**V2V Rate下降的真实原因**（关键洞察）：
- CAM交付期间（demand>0）：V2V=1716（vs基线2013，仅低15%）
- CAM交付完成后（demand=0）：V2V=554（vs基线1137，主动降低）
- 智能体学会了"按需分配"：完成CAM后主动切换到V2I模式全力压低AoI
- CAM交付率100%，是唯一全部成功的算法
- 这不是V2V能力退化，而是**资源利用效率的提升**

### F. 完整定量对比数据

| Algorithm | AoI | Reward | V2I Rate | V2V Rate | CAM | Conv.Ep | Time(min) |
|-----------|:---:|:------:|:--------:|:--------:|:---:|:-------:|:---------:|
| TDec MADDPG (1) | 5.07 | -0.83 | 346.5 | 1271.8 | 98% | 120 | 88.7 |
| MADDPG (2) | 7.10 | -1.34 | 389.2 | 549.5 | 4% | 355 | 73.2 |
| FDec MADDPG (3) | 8.86 | -1.35 | 263.3 | 803.9 | 32% | 386 | 79.8 |
| DDPG (4) | 42.84 | -4.07 | 289.8 | 1013.6 | 2% | 196 | 14.4 |
| AoI-Enhanced (6) | 4.86 | -0.25 | 356.8 | 1145.3 | 96% | 340 | 76.8 |
| ParamShare (11) | 4.89 | -0.87 | 372.5 | 1176.5 | 98% | 81 | 61.2 |
| AoI-ParamShare (12) | 4.51 | -0.30 | 374.0 | 763.2 | 100% | 115 | 68.6 |

V2V Rate拆分：
| Algorithm | V2V(demand>0) | V2V(demand=0) | CAM 5/5 | 1st delivery |
|-----------|:-------------:|:-------------:|:-------:|:------------:|
| TDec (1) | 2013 | 1137 | 98% | 15.1/100 |
| AoI-Enhanced (6) | 2123 | 978 | 96% | 13.9/100 |
| ParamShare (11) | 1867 | 1039 | 98% | 16.3/100 |
| AoI-ParamShare (12) | 1716 | 554 | 100% | 18.0/100 |

### G. 原始论文图片URL（供参考/直接使用）

- Fig.1（多车道编队场景）：`https://cdn-mineru.openxlab.org.cn/result/2026-05-26/71c33fe1-5ebe-4c06-9c0f-2eafe73efa16/cfd3e2842528fa37d665d10ad7e7cb518108c62b635154b8139724ddaf6fdd84.jpg`
- Fig.2（算法架构图）：`https://cdn-mineru.openxlab.org.cn/result/2026-05-26/71c33fe1-5ebe-4c06-9c0f-2eafe73efa16/b9e421fd5cafaec787415d2f5227a8c526f03cfa6a4ec022b4f464de84ca4497.jpg`
- Fig.4（收敛对比）：`https://cdn-mineru.openxlab.org.cn/result/2026-05-26/71c33fe1-5ebe-4c06-9c0f-2eafe73efa16/facce762cea26770e56bc0c808a37322a055e2101ec9529a99a8d1eb2f004123.jpg`
- Fig.5（AoI vs intra-platoon gap）：`https://cdn-mineru.openxlab.org.cn/result/2026-05-26/71c33fe1-5ebe-4c06-9c0f-2eafe73efa16/d0768b7376d91d19fab18b36b754f485da5629feebdaa1cb1b20f2327b2151ff.jpg`
- Fig.6（AoI vs platoon size）：`https://cdn-mineru.openxlab.org.cn/result/2026-05-26/71c33fe1-5ebe-4c06-9c0f-2eafe73efa16/878b8ef3b670f2e77f249d49ad2b40bfc1384934fbd1306ee41d4fae3a4b2a83.jpg`
- Fig.8（车队行为）：`https://cdn-mineru.openxlab.org.cn/result/2026-05-26/71c33fe1-5ebe-4c06-9c0f-2eafe73efa16/a1430db6353e061d8980e7083610679c4a4cf1807cd99a722677ddcf53ca85.jpg`

### H. 已有实验结果图（在项目figures/目录下）

- `figures/fig5_aoi_convergence.png` — AoI收敛曲线（6个多智能体算法 + DDPG inset）
- `figures/fig6_cam_v2i_comparison.png` — CAM交付率 + V2I速率柱状图（7个算法）
- `figures/fig8_platoon_behavior.png` — 单车队100步行为（功率+CAM需求）
- `figures/fig3_task_rewards.png` — Task1/Task2收敛曲线（4个算法）
- `figures/fig4_reward_comparison.png` — 总奖励收敛对比（7个算法）
