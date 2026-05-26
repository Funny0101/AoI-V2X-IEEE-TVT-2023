"""Fig 5: AoI convergence comparison — main algorithms + DDPG inset."""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12

def smooth(data, w):
    out = np.copy(data)
    for i in range(len(data)):
        start = max(0, i - w + 1)
        out[i] = np.mean(data[start:i+1])
    return out

algos_main = [
    ('1-Modified MADDPG with TDec/model/marl_model/',             'split',   'TDec MADDPG',     '#1f77b4', '-'),
    ('2-Modified MADDPG/model/marl_model/',                        'single',  'MADDPG',          '#ff7f0e', '-'),
    ('3-MADDPG_FDec/model/marl_model/',                            'single',  'FDec MADDPG',     '#2ca02c', '-'),
    ('6-Modified MADDPG with AoI-Enhanced/model/marl_model/',      'split',   'AoI-Enhanced',    '#9467bd', '--'),
    ('11-Modified MADDPG with ParamShare/model/marl_model/',       'split',   'ParamShare',      '#8c564b', '--'),
    ('12-Modified MADDPG with AoI-ParamShare/model/marl_model/',   'split',   'AoI-ParamShare',  '#e6550d', '--'),
]

fig, (ax_main, ax_ddpg) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [3, 1]})

for folder, rtype, label, color, ls in algos_main:
    aoi = scipy.io.loadmat(folder + 'AoI.mat')['AoI']
    mean_aoi = np.mean(aoi, axis=0)
    smoothed = smooth(mean_aoi, 20)
    ax_main.plot(smoothed, label=label, color=color, linestyle=ls, linewidth=1.5)

ax_main.set_xlabel('Episode', fontsize=13)
ax_main.set_ylabel('Average AoI', fontsize=13)
ax_main.set_title('AoI Convergence (Multi-Agent Algorithms)', fontsize=13, fontweight='bold')
ax_main.legend(fontsize=9, loc='upper right')
ax_main.grid(True, alpha=0.3)
ax_main.set_xlim(0, 499)
ax_main.set_ylim(0, 20)

ddpg_folder = '4-DDPG/model/marl_model/'
ddpg_aoi = scipy.io.loadmat(ddpg_folder + 'AoI.mat')['AoI']
ddpg_mean = np.mean(ddpg_aoi, axis=0)
ddpg_smooth = smooth(ddpg_mean, 20)
ax_ddpg.plot(ddpg_smooth, color='#d62728', linewidth=1.5)
ax_ddpg.set_title('DDPG', fontsize=13, fontweight='bold')
ax_ddpg.set_xlabel('Episode', fontsize=12)
ax_ddpg.set_ylabel('Average AoI', fontsize=12)
ax_ddpg.grid(True, alpha=0.3)
ax_ddpg.set_xlim(0, 499)

fig.tight_layout()
fig.savefig('figures/fig5_aoi_convergence.png', dpi=300, bbox_inches='tight')
print('Saved figures/fig5_aoi_convergence.png')
