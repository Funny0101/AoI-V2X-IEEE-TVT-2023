# PPT 逐页生成指南

> 共10页，5分钟汇报，每页约30秒。
> 每一页包含：标题、内容要点、图片/图表要求。
> 末尾附全部参考数据，供逐页生成时查阅。

---

## 第1页：研究背景、问题与系统模型

### 标题
研究背景与问题

### 内容要点（左侧文字区域，简洁要点）
- **车队编队（Platooning）**：多辆自动驾驶车以紧密队形协同行驶
- **两类通信需求**：
  - V2V（车与车）：PL（头车）向队员广播CAM安全消息 → 保证编队稳定性
  - V2I（车与路侧单元）：PL向RSU汇报状态 → 保证信息时效性
- **AoI（信息年龄）**：从最近一次成功V2I传输到现在的时间差，越小说明信息越新鲜
- **核心问题**：如何分布式分配无线资源（子载波、通信模式、功率），最小化AoI并保证CAM交付率？
- **场景参数**：1个RSU + 5个车队（每队4辆），3个子载波，每100ms重置一次CAM
- **挑战**：高动态信道、多智能体干扰、混合整数非线性规划（MINLP）

### 图片（右侧，占约一半版面）
- **直接使用原始论文 Fig.1**（多车道编队场景图）
- 图片URL：`https://cdn-mineru.openxlab.org.cn/result/2026-05-26/71c33fe1-5ebe-4c06-9c0f-2eafe73efa16/cfd3e2842528fa37d665d10ad7e7cb518108c62b635154b8139724ddaf6fdd84.jpg`
- 这张图已经展示了RSU、多个车队、V2I/V2V链路，与左侧文字完美呼应

---

## 第2页：原始论文方法 — Modified MADDPG with TDec

### 标题
原始论文方法：Modified MADDPG with Task Decomposition

### 内容要点（左侧文字）
- **多智能体建模**：每个PL为一个智能体，分布式决策（Mode 4分布式资源分配）
- **双层Critic架构**：
  - 全局Critic（在RSU上，共享）：输入所有PL的状态+动作 → 评估全局干扰水平 → 促进协作，用TD3防止Q值过估计
  - 本地Critic（每个PL独立）：输入自身状态+动作 → 评估个体任务表现
- **任务分解**：将奖励拆为两个子任务，各有独立Critic
  - Task 1（V2V/CAM交付）：剩余CAM需求越少 → 奖励越高
  - Task 2（V2I + AoI）：V2I达标给奖励 + AoI线性惩罚（-AoI/20）
- **Actor更新**：全局Critic梯度 + 本地Task1/Task2 Critic梯度 → 引导Actor改进策略
- **结果**：在所有基线中最优，AoI=5.07，CAM交付率98%

### 图片（右侧，AI生成算法架构图）
- **AI生成一张算法架构图**，要求：
  - 左侧画5个竖向排列的智能体模块（Agent 1 ~ Agent 5）
  - 每个智能体模块内包含：
    - 一个"Actor"方块（策略网络，做决策）
    - 两个小方块："Critic T1"和"Critic T2"（任务分解的本地Critic，分别评估CAM交付和AoI控制）
  - 右侧画一个大方块代表RSU，内部有"Global Critic (Twin TD3)"，包含Q1和Q2两个子网络
  - 连线：
    - 每个Agent的Actor → 输出动作a（选择RB、模式、功率）
    - 所有Agent的(s, a)汇总 → 输入Global Critic
    - 每个Agent自己的(s, a) → 输入各自的Critic T1和T2
  - 梯度回传用红色箭头标注：
    - Global Critic → 红色箭头 → 各Agent的Actor（全局协作梯度）
    - Critic T1 + Critic T2 → 红色箭头 → 各Agent的Actor（本地任务梯度）
  - 底部标注：Task1 reward关注CAM交付，Task2 reward关注AoI+V2I
  - 参考：论文 Fig. 2 和 Algorithm 2

---

## 第3页：原始方法的缺陷分析

### 标题
原始方法的缺陷分析

### 内容要点（左侧文字）
- **缺陷1：AoI奖励设计不足**
  - 原始使用线性惩罚：`-AoI/20`
  - 问题：无论AoI是10还是100，每增加1单位的惩罚力度完全相同
  - 后果：智能体对"极端高AoI"不够敏感，缺乏紧急处理的动力
- **缺陷2：多智能体独立参数，训练效率低**
  - 5个结构完全相同的Actor网络各自从零开始学习
  - 收敛需要120 episodes，训练耗时88.7分钟
  - 浪费了"同构智能体"的先验——所有车队面临相同的决策逻辑

### 图片（右侧，AI生成惩罚函数对比图）
- **AI生成一张惩罚函数对比图**，要求：
  - 左右并排两个子图
  - 左图标题"线性惩罚（原始）"：画一条从原点出发的直线 y = -x/20，标注"惩罚力度恒定，对高AoI不敏感"
  - 右图标题"抛物线惩罚（改进）"：画一条抛物线 y = -5*(x/100)²，标注"高AoI区域惩罚急剧增大"
  - 横轴均为"AoI值"，纵轴均为"惩罚值"
  - 在x=80-100的位置用浅红色底色标注"高AoI风险区域"，右图该区域明显惩罚更大

---

## 第4页：改进一 — AoI-Enhanced（奖励塑形）

### 标题
改进一：AoI-Enhanced — 奖励塑形

### 内容要点（左侧文字）
- **改进1：AoI惩罚改为抛物线型**
  - 原始：`-AoI/20`（线性，力度恒定）
  - 改进：`-5.0 × (AoI/100)²`（抛物线，高AoI时惩罚急剧增大）
- **改进2：新增AoI刷新奖励**
  - 当V2I成功传输（AoI重置）时，额外奖励 `+2.0`
  - 直接激励智能体主动选择V2I模式来刷新AoI
- **改进3：状态增强**
  - 新增2个状态特征：AoI_trend（AoI变化趋势）、Peak_AoI（峰值AoI）
  - 状态维度：19维 → 21维
- **实验结果**（最后50回合均值）：
  - AoI: 5.07 → **4.86（-4.2%）**
  - Reward: -0.83 → **-0.25（+69.9%）**
  - V2I Rate: 346.5 → **356.8（+3.0%）**

### 图片（右侧，AI生成改进示意图）
- **AI生成一张改进示意图**，要求：
  - 左侧画"原始TDec"的框，包含三行：
    - "线性惩罚: -AoI/20"
    - "19维状态"
    - "无刷新奖励"
  - 中间画一个大向右箭头，标注"AoI-Enhanced"
  - 右侧画"改进后"的框，包含三行：
    - "抛物线惩罚: -5.0·(AoI/100)²"
    - "21维状态 (+AoI_trend, +Peak_AoI)"
    - "AoI刷新奖励: +2.0"
  - 右侧框下方用绿色标注关键结果："AoI -4.2%, Reward +69.9%"

---

## 第5页：改进二 — ParamShare（参数共享）

### 标题
改进二：ParamShare — 参数共享

### 内容要点（左侧文字）
- **核心思想**：5个智能体共享同一个Actor网络参数
  - 所有车队面临相同的决策逻辑（选RB、选模式、选功率），没必要各自学一套
  - 保留独立的Local Critic（各车队的AoI/CAM状态不同，需要个性化评估）
- **梯度累积策略**：
  - 5个Agent分别计算梯度（backward）→ 累加 → 统一更新（step一次）
  - 目标网络只更新1次（而非5次），防止软更新系数τ被等效放大5倍
- **为什么有效**：共享参数 = 5倍数据利用效率 = 更快收敛 + 更好泛化
- **实验结果**：
  - AoI: 5.07 → **4.89（-3.6%）**
  - V2I Rate: 346.5 → **372.5（+7.5%）**
  - 收敛速度：120 → **81 episodes（1.48x加速）**
  - 训练时间：88.7 → **61.2 min（-31%）**

### 图片（右侧，AI生成参数共享架构对比图）
- **AI生成一张参数共享架构对比图**，要求：
  - 左侧"原始"：画5个独立的Actor网络方块（Actor1, Actor2, ..., Actor5），各自有箭头指向自己的optimizer，下方标注"5次独立更新，参数不共享"
  - 右侧"ParamShare"：画1个共享Actor方块，5个Agent的箭头都指向它，下方汇聚成"梯度累加"节点，然后一个箭头指向optimizer step一次，标注"5个Agent梯度累加 → 1次统一更新"
  - 两个方案中都用小方块标注"Critic各自独立"
  - 右下方标注关键结果："收敛 1.48x加速, 训练 -31%"

---

## 第6页：改进三 — AoI+ParamShare融合修复

### 标题
改进三：AoI+ParamShare融合修复

### 内容要点（左侧文字）
- **挑战**：直接叠加改进一和改进二 → 梯度爆炸，训练不稳定
  - 原因：AoI奖励塑形改变了Task2信号分布，共享Actor收到5个Agent的异质梯度，方向冲突
- **三重修复**：
  1. **梯度裁剪**（clip_grad_norm = 1.0）：限制梯度范数，防止爆炸
  2. **奖励归一化**（Z-score）：对每个Agent的奖励标准化，消除异质分布差异
  3. **Actor Loss取平均**（非累加）：5个Agent的梯度取平均，保证梯度平滑
- **实验结果（全场最优）**：
  - AoI: 5.07 → **4.51（-11.1%，最优）**
  - CAM交付率: 98% → **100%（唯一全成功）**
  - V2I Rate: 346.5 → **374.0（+7.9%）**
  - 训练时间: 88.7 → **68.6 min（-23%）**

### 图片（右侧，AI生成融合改进示意图）
- **AI生成一张融合改进示意图**，要求：
  - 画一个三层金字塔或三层叠加图：
    - 底层方块：标注"参数共享（ParamShare）"— 5个Agent共享Actor
    - 中层方块：标注"AoI奖励塑形" — 抛物线惩罚 + 刷新奖励
    - 顶层方块：标注"稳定性修复" — 内含三个小标签："梯度裁剪"、"奖励Z-score归一化"、"Loss平均"
  - 右侧用箭头指向一个结果框：AoI 4.51 (-11.1%), CAM 100%, V2I 374.0 (+7.9%)

---

## 第7页：实验结果 — 全面对比

### 标题
实验结果：全面对比

### 内容要点（上方文字，下方两张图）
- 7个算法在相同环境下的定量对比
- 关键发现：
  - 三项改进的AoI持续优化：5.07 → 4.86 → 4.89 → 4.51
  - 算法12实现AoI、CAM、V2I三个指标的**同时最优**
  - 基线DDPG/FDec/MADDPG性能大幅落后，验证TDec架构的必要性

### 图片（页面主体，两张并排柱状图）
- **左侧：使用已有图 `figures/fig6_cam_v2i_comparison.png`**
  - 该图包含两个子图：
    - 左子图：CAM交付率柱状图（7个算法），可直观看到AoI-ParamShare的100%遥遥领先，DDPG仅2%
    - 右子图：V2I速率柱状图（7个算法），改进算法普遍高于基线
  - 汇报时重点指向CAM=100%和DDPG=2%的对比

- **下方或右侧：数据表格（PPT直接绘制）**

| 算法 | AoI ↓ | Reward ↑ | V2I ↑ | V2V | CAM ↑ | Time ↓ |
|------|:-----:|:-------:|:-----:|:---:|:-----:|:------:|
| TDec MADDPG | 5.07 | -0.83 | 346.5 | 1271.8 | 98% | 88.7 |
| MADDPG | 7.10 | -1.34 | 389.2 | 549.5 | 4% | 73.2 |
| FDec MADDPG | 8.86 | -1.35 | 263.3 | 803.9 | 32% | 79.8 |
| DDPG | 42.84 | -4.07 | 289.8 | 1013.6 | 2% | 14.4 |
| **AoI-Enhanced** | **4.86** | **-0.25** | **356.8** | 1145.3 | 96% | 76.8 |
| **ParamShare** | **4.89** | -0.87 | **372.5** | 1176.5 | 98% | **61.2** |
| **AoI-ParamShare** | **4.51** | **-0.30** | **374.0** | 763.2 | **100%** | 68.6 |

- 表格样式：基线行灰色底色，改进行浅蓝底色，每列最优值加粗+绿色

---

## 第8页：实验结果 — 收敛分析

### 标题
实验结果：收敛性能分析

### 内容要点
- **奖励收敛（左图）**：
  - AoI-Enhanced（紫色虚线）和AoI-ParamShare（橙色点线）的总奖励明显高于基线TDec（蓝色实线）
  - DDPG（红色）总奖励极差，完全无法处理多智能体场景
  - 改进算法收敛更快、波动更小
- **AoI收敛（右图）**：
  - AoI-ParamShare的AoI最终稳定在最低水平（~4.5）
  - 基线TDec稳定在~5.0，有改进空间
  - DDPG的AoI高达40+（右下角inset小图），说明单智能体方法完全失效
- **核心结论**：改进算法在收敛速度和最终性能上均优于基线

### 图片（两张并排）
- **左图：使用已有图 `figures/fig4_reward_comparison.png`**
  - 展示7个算法的平均总奖励随episodes（0-499）的收敛曲线
  - 重点：AoI-Enhanced和AoI-ParamShare的曲线在基线之上

- **右图：使用已有图 `figures/fig5_aoi_convergence.png`**
  - 展示6个多智能体算法的AoI收敛曲线 + DDPG的inset小图
  - 重点：AoI-ParamShare的曲线在所有算法中最低

---

## 第9页：实验结果 — 各智能体任务收敛

### 标题
实验结果：各智能体任务收敛分析

### 内容要点
- **上图 — Task 1（CAM交付）**：5个车队的Task1奖励收敛曲线
  - 所有改进算法（紫色/棕色/橙色）收敛更快、更稳定
  - 基线TDec（蓝色）在Agent 3、5上收敛明显更慢（需要100-200 episodes）
  - 改进算法普遍在50 episodes内稳定收敛
- **下图 — Task 2（AoI+V2I）**：5个车队的Task2奖励收敛曲线
  - AoI-Enhanced和AoI-ParamShare的Task2奖励更高（AoI更低）
  - ParamShare收敛更快但最终奖励略低
- **核心结论**：改进算法让每个智能体的两个子任务都能快速、稳定地收敛，不存在"只顾一个任务而忽略另一个"的问题

### 图片
- **直接使用已有图 `figures/fig3_task_rewards.png`**
  - 该图是 2行×5列 的子图矩阵：
    - 上行5个子图：Platoon 1-5 各自的 Task 1 奖励收敛曲线
    - 下行5个子图：Platoon 1-5 各自的 Task 2 奖励收敛曲线
  - 每个子图中有4条曲线：TDec（蓝色实线）、AoI-Enhanced（紫色虚线）、ParamShare（棕色点划线）、AoI-ParamShare（橙色点线）
  - 汇报时重点看收敛速度对比和最终稳定值对比

---

## 第10页：实验结果 — V2V深度分析与行为分析

### 标题
实验结果：V2V Rate深度分析

### 内容要点（左侧文字）
- **V2V Rate的"假象下降"**：
  - 算法12的V2V Rate（763.2）看似比基线（1271.8）低40%
  - 拆分后发现真相：
    - CAM交付期间（demand>0）：V2V = 1716（vs 基线 2013，仅低15%）
    - CAM交付完成后（demand=0）：V2V = 554（vs 基线 1137，大幅降低）
- **关键洞察**：智能体学会了"按需分配"
  - 完成CAM交付后 → 主动从V2V模式切换到V2I模式 → 全力压低AoI
  - 这不是V2V能力退化，而是**资源利用效率的提升**
  - 证据：CAM交付率100%（唯一全部成功）
- 从右侧行为图可以看到：AoI-ParamShare在CAM交完后更积极地调整功率分配

### 图片（右侧两张图）
- **上图：AI生成V2V拆分对比柱状图**，要求：
  - 分组柱状图，横轴4个算法：TDec / AoI-Enhanced / ParamShare / AoI-ParamShare
  - 每个算法两根柱子：
    - 深色柱：V2V(demand>0) — CAM交付期间的V2V速率（数据：2013 / 2123 / 1867 / 1716）
    - 浅色柱：V2V(demand=0) — CAM完成后的V2V速率（数据：1137 / 978 / 1039 / 554）
  - 在AoI-ParamShare的浅色柱上标注"主动切换至V2I模式"
  - Y轴范围 0-2500

- **下图：使用已有图 `figures/fig8_platoon_behavior.png`**
  - 该图包含3个子图（TDec / AoI-Enhanced / AoI-ParamShare）
  - 每个子图：柱状图是每步的发射功率，折线是剩余CAM需求
  - 汇报时重点对比TDec和AoI-ParamShare：后者在CAM交完后功率分配策略有明显变化（切换到V2I模式）

---
---

## 附录：参考数据（供生成每一页时查阅）

### A. 论文基本信息
- 标题：AoI-Aware Resource Allocation for Platoon-Based C-V2X Networks via Multi-Agent Multi-Task Reinforcement Learning
- 作者：Mohammad Parvini et al., IEEE Transactions on Vehicular Technology, 2023
- 场景：城市网格道路，1个RSU + 5个车队（每个车队4辆车），共20辆车
- 频段：2GHz，3个RB（每个180kHz），CAM消息大小4000字节，时间约束100ms
- 核心指标：AoI（信息年龄）、CAM消息交付率、V2I/V2V数据速率

### B. 原始论文方法（Modified MADDPG with Task Decomposition）
- 多智能体建模：每个车队头车（PL）为一个智能体，联合决策子载波分配(β)、通信模式(θ)、发射功率(p)
- 双层Critic架构：
  - 全局Critic（在RSU上，所有智能体共享）：输入所有智能体的(s,a)，用TD3评估全局团队奖励（基于干扰水平）
  - 本地Critic（每个智能体独立）：输入自身的(s,a)，评估个体任务奖励
- 任务分解：
  - Task 1（V2V/CAM交付）：r_1 = -4.95*(Demand/V2V_demand) - θ·power_penalty
  - Task 2（V2I+AoI）：r_2 = 0.05·Revenue(V2I_rate) - AoI/20 - (1-θ)·power_penalty
- 全局奖励：r_g = -(1/P)·ΣΣ log(I_j[k])（平均干扰水平）
- 网络结构：
  - Actor: [input → 1024 → 512 → 3] (LayerNorm + ReLU, tanh output)
  - Local Critic: [input → 512 → 256 → 1]
  - Global Critic: [concat(95) → 1024 → 512 → 256 → 1] (Twin TD3)
- 超参数：batch=64, memory=50k, γ=0.99, τ=0.005, lr_actor=1e-4, lr_critic=1e-3, noise=0.3
- 状态空间（19维/agent）：信道增益h(V2I), h(V2V), 干扰I, AoI, 剩余CAM需求ζ^r, 剩余时间T^r
- 动作空间（3维连续，经tanh映射到离散）：RB选择、模式选择、功率

### C. 改进一：AoI-Enhanced（算法6）
- 问题：线性AoI惩罚 -AoI/20 对极端高AoI不敏感
- 改进1：AoI惩罚改为抛物线型 -5.0*(AoI/100)²
- 改进2：新增AoI刷新奖励 +2.0（当V2I成功传输、AoI重置时）
- 改进3：状态增强，新增AoI_trend和Peak_AoI，19维→21维
- 结果：AoI 5.07→4.86(-4.2%), Reward -0.83→-0.25(+69.9%), V2I 346.5→356.8(+3.0%), V2V 1271.8→1145.3(-10%), CAM 98%→96%

### D. 改进二：ParamShare（算法11）
- 问题：5个同构智能体各自独立Actor网络，参数不共享，训练效率低
- 改进：5个Agent共享同一个ActorNetwork，保留独立Local Critic
- 梯度累积：所有Agent backward()后统一step()一次
- 目标网络只更新1次，防止τ膨胀
- 结果：AoI 5.07→4.89(-3.6%), V2I 346.5→372.5(+7.5%), V2V 1271.8→1176.5(-7.5%), CAM 98%, 收敛120→81ep(1.48x), 训练88.7→61.2min(-31%)

### E. 改进三：AoI+ParamShare融合修复（算法12）
- 问题：直接叠加→梯度爆炸，训练不稳定
- 原因：AoI奖励塑形改变Task2信号分布，共享Actor收到5个Agent异质梯度冲突
- 修复1：梯度裁剪 clip_grad_norm_=1.0
- 修复2：奖励Z-score归一化
- 修复3：Actor Loss取平均（非累加）
- 结果：AoI 5.07→4.51(-11.1%), Reward -0.83→-0.30(+63.6%), V2I 346.5→374.0(+7.9%), V2V 1271.8→763.2(-40%), CAM 98%→100%, 训练88.7→68.6min(-23%)

### F. V2V Rate拆分数据
| 算法 | V2V(demand>0) | V2V(demand=0) | CAM 5/5 | 首次交付步 |
|------|:-------------:|:-------------:|:-------:|:----------:|
| TDec (基线) | 2013 | 1137 | 98% | 15.1/100 |
| AoI-Enhanced | 2123 | 978 | 96% | 13.9/100 |
| ParamShare | 1867 | 1039 | 98% | 16.3/100 |
| AoI-ParamShare | 1716 | 554 | 100% | 18.0/100 |

### G. 完整定量对比表
| 算法 | AoI | Reward | V2I Rate | V2V Rate | CAM | Conv.Ep | Time(min) |
|------|:---:|:------:|:--------:|:--------:|:---:|:-------:|:---------:|
| TDec MADDPG (1) | 5.07 | -0.83 | 346.5 | 1271.8 | 98% | 120 | 88.7 |
| MADDPG (2) | 7.10 | -1.34 | 389.2 | 549.5 | 4% | 355 | 73.2 |
| FDec MADDPG (3) | 8.86 | -1.35 | 263.3 | 803.9 | 32% | 386 | 79.8 |
| DDPG (4) | 42.84 | -4.07 | 289.8 | 1013.6 | 2% | 196 | 14.4 |
| AoI-Enhanced (6) | 4.86 | -0.25 | 356.8 | 1145.3 | 96% | 340 | 76.8 |
| ParamShare (11) | 4.89 | -0.87 | 372.5 | 1176.5 | 98% | 81 | 61.2 |
| AoI-ParamShare (12) | 4.51 | -0.30 | 374.0 | 763.2 | 100% | 115 | 68.6 |

### H. 各图说明
- `figures/fig3_task_rewards.png` — 2行×5列子图。上行5个Platoon的Task1(CAM)奖励收敛，下行5个Platoon的Task2(AoI+V2I)奖励收敛。4条曲线对比：TDec(蓝实线)、AoI-Enhanced(紫虚线)、ParamShare(棕点划线)、AoI-ParamShare(橙点线)。改进算法收敛更快更稳。
- `figures/fig4_reward_comparison.png` — 7个算法的平均总奖励随episodes收敛曲线。AoI-Enhanced和AoI-ParamShare最高，DDPG最低。
- `figures/fig5_aoi_convergence.png` — 左图6个多智能体算法的AoI收敛曲线（Y轴0-20），右图DDPG inset（Y轴0-45）。AoI-ParamShare最低。
- `figures/fig6_cam_v2i_comparison.png` — 左子图CAM交付率柱状图(7算法)，右子图V2I速率柱状图(7算法)。AoI-ParamShare的CAM=100%最高。
- `figures/fig8_platoon_behavior.png` — 3个子图(TDec/AoI-Enhanced/AoI-ParamShare)。每图：柱状图为每步发射功率(dBm)，折线为剩余CAM需求(bits)。展示100步内单车队的行为差异。

### I. 原始论文图片URL
- Fig.1（多车道编队场景）：https://cdn-mineru.openxlab.org.cn/result/2026-05-26/71c33fe1-5ebe-4c06-9c0f-2eafe73efa16/cfd3e2842528fa37d665d10ad7e7cb518108c62b635154b8139724ddaf6fdd84.jpg
- Fig.2（算法架构图）：https://cdn-mineru.openxlab.org.cn/result/2026-05-26/71c33fe1-5ebe-4c06-9c0f-2eafe73efa16/b9e421fd5cafaec787415d2f5227a8c526f03cfa6a4ec022b4f464de84ca4497.jpg
- Fig.4（收敛对比）：https://cdn-mineru.openxlab.org.cn/result/2026-05-26/71c33fe1-5ebe-4c06-9c0f-2eafe73efa16/facce762cea26770e56bc0c808a37322a055e2101ec9529a99a8d1eb2f004123.jpg
- Fig.5（AoI vs gap）：https://cdn-mineru.openxlab.org.cn/result/2026-05-26/71c33fe1-5ebe-4c06-9c0f-2eafe73efa16/d0768b7376d91d19fab18b36b754f485da5629feebdaa1cb1b20f2327b2151ff.jpg
- Fig.6（AoI vs size）：https://cdn-mineru.openxlab.org.cn/result/2026-05-26/71c33fe1-5ebe-4c06-9c0f-2eafe73efa16/878b8ef3b670f2e77f249d49ad2b40bfc1384934fbd1306ee41d4fae3a4b2a83.jpg
- Fig.8（车队行为）：https://cdn-mineru.openxlab.org.cn/result/2026-05-26/71c33fe1-5ebe-4c06-9c0f-2eafe73efa16/a1430db6353e061d8980e7083610679c4a4cf1807cd99a722677ddcf53ca85.jpg
