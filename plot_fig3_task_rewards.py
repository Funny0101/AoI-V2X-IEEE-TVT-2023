"""Fig 3: Task 1 and Task 2 reward convergence per agent (TDec + AoI-Enhanced + ParamShare)."""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

def smooth(data, w):
    out = np.copy(data)
    for i in range(len(data)):
        start = max(0, i - w + 1)
        out[i] = np.mean(data[start:i+1])
    return out

algos = [
    ('1-Modified MADDPG with TDec/model/marl_model/', 'TDec',         '#1f77b4', '-'),
    ('6-Modified MADDPG with AoI-Enhanced/model/marl_model/', 'AoI-Enhanced', '#9467bd', '--'),
    ('11-Modified MADDPG with ParamShare/model/marl_model/', 'ParamShare',  '#8c564b', '-.'),
]

data = {}
for folder, label, _, _ in algos:
    data[label] = {
        't1': scipy.io.loadmat(folder + 'reward_t1.mat')['reward_t1'],
        't2': scipy.io.loadmat(folder + 'reward_t2.mat')['reward_t2'],
    }

n_platoon = 5
n_episode = 500
window = 20

fig, axes = plt.subplots(2, n_platoon, figsize=(20, 7), sharex=True)

for i in range(n_platoon):
    axes[0, i].set_title(f'Platoon {i+1}', fontsize=12, fontweight='bold')
    for folder, label, color, ls in algos:
        t1_s = smooth(data[label]['t1'][i], window)
        t2_s = smooth(data[label]['t2'][i], window)
        axes[0, i].plot(t1_s, color=color, linestyle=ls, linewidth=1.3)
        axes[1, i].plot(t2_s, color=color, linestyle=ls, linewidth=1.3)

    axes[0, i].set_ylim(-6, 1)
    axes[0, i].grid(True, alpha=0.3)
    axes[1, i].set_ylim(-6, 1)
    axes[1, i].grid(True, alpha=0.3)
    axes[1, i].set_xlabel('Episode')

axes[0, 0].set_ylabel('Task 1 Reward')
axes[1, 0].set_ylabel('Task 2 Reward')
axes[0, 0].set_xlim(0, n_episode - 1)

legend_elements = [Line2D([0], [0], color=c, linewidth=1.5, linestyle=ls, label=l)
                   for _, l, c, ls in algos]
axes[0, n_platoon - 1].legend(handles=legend_elements, loc='lower right', fontsize=9)

fig.suptitle('Task 1 and Task 2 Reward Convergence per Platoon', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('figures/fig3_task_rewards.png', dpi=300, bbox_inches='tight')
print('Saved figures/fig3_task_rewards.png')
