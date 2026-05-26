"""Fig 4: Average total reward comparison across all algorithms."""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12

window = 20

def smooth(data, w):
    out = np.copy(data)
    for i in range(len(data)):
        start = max(0, i - w + 1)
        out[i] = np.mean(data[start:i+1])
    return out

def load_total_reward(folder, reward_type):
    if reward_type == 'split':
        t1 = scipy.io.loadmat(folder + 'reward_t1.mat')['reward_t1']
        t2 = scipy.io.loadmat(folder + 'reward_t2.mat')['reward_t2']
        return t1 + t2
    else:
        d = scipy.io.loadmat(folder + 'reward.mat')
        key = [k for k in d.keys() if not k.startswith('_')][0]
        return d[key]

algos = [
    ('1-Modified MADDPG with TDec/model/marl_model/',             'split',   'TDec MADDPG',     '#1f77b4', '-'),
    ('2-Modified MADDPG/model/marl_model/',                        'single',  'MADDPG',          '#ff7f0e', '-'),
    ('3-MADDPG_FDec/model/marl_model/',                            'single',  'FDec MADDPG',     '#2ca02c', '-'),
    ('4-DDPG/model/marl_model/',                                   'single',  'DDPG',            '#d62728', '-'),
    ('6-Modified MADDPG with AoI-Enhanced/model/marl_model/',      'split',   'AoI-Enhanced',    '#9467bd', '--'),
    ('11-Modified MADDPG with ParamShare/model/marl_model/',       'split',   'ParamShare',      '#8c564b', '--'),
    ('12-Modified MADDPG with AoI-ParamShare/model/marl_model/',   'split',   'AoI-ParamShare',  '#e6550d', '--'),
]

fig, ax = plt.subplots(figsize=(10, 6))

for folder, rtype, label, color, ls in algos:
    total = load_total_reward(folder, rtype)
    mean_r = np.mean(total, axis=0)
    smoothed = smooth(mean_r, window)
    ax.plot(smoothed, label=label, color=color, linestyle=ls, linewidth=1.5)

ax.set_xlabel('Episode', fontsize=13)
ax.set_ylabel('Average Total Reward', fontsize=13)
ax.set_title('Average Total Reward Convergence', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 499)

fig.tight_layout()
fig.savefig('figures/fig4_reward_comparison.png', dpi=300, bbox_inches='tight')
print('Saved figures/fig4_reward_comparison.png')
