"""Fig 8: Single platoon 100-step behavior — our 3 improvements vs baseline."""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

algos = [
    ('1-Modified MADDPG with TDec/model/marl_model/', 'TDec', '#1f77b4'),
    ('6-Modified MADDPG with AoI-Enhanced/model/marl_model/', 'AoI-Enhanced', '#9467bd'),
    ('12-Modified MADDPG with AoI-ParamShare/model/marl_model/', 'AoI-ParamShare', '#e6550d'),
]

platoon_idx = 0
ep_idx = -1

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax_idx, (folder, label, color) in enumerate(algos):
    power = scipy.io.loadmat(folder + 'power.mat')['power']
    demand = scipy.io.loadmat(folder + 'demand.mat')['demand']
    aoi_evo = scipy.io.loadmat(folder + 'AoI_evolution.mat')['AoI_evolution']

    p = power[platoon_idx, ep_idx, :]
    d = demand[platoon_idx, ep_idx, :]
    steps = np.arange(len(p))

    ax1 = axes[ax_idx]
    ax1.bar(steps, p, color=color, alpha=0.5, width=0.8)
    ax1.set_xlabel('Time Step (ms)')
    ax1.set_ylabel('Transmit Power (dBm)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 35)
    ax1.set_title(label, fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(steps, d, color='#d62728', linewidth=1.5, marker='o', markersize=2)
    ax2.set_ylabel('V2V Demand Remaining (bits)', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [Patch(facecolor=color, alpha=0.5, label='Power'),
                       Line2D([0], [0], color='#d62728', linewidth=1.5, label='Demand')]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)

fig.suptitle('Single Platoon Behavior over 100 Steps', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('figures/fig8_platoon_behavior.png', dpi=300, bbox_inches='tight')
print('Saved figures/fig8_platoon_behavior.png')
