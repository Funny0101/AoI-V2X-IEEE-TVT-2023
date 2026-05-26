"""Fig 6: CAM delivery rate and V2I rate comparison — two subplots."""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12

def compute_cam(demand):
    last50 = demand[:, -50:, :]
    return np.mean(np.sum(last50[:,:,-1] <= 0, axis=0) == 5) * 100

algos = [
    ('1-Modified MADDPG with TDec/model/marl_model/', 'TDec', '#1f77b4'),
    ('2-Modified MADDPG/model/marl_model/', 'MADDPG', '#ff7f0e'),
    ('3-MADDPG_FDec/model/marl_model/', 'FDec', '#2ca02c'),
    ('4-DDPG/model/marl_model/', 'DDPG', '#d62728'),
    ('6-Modified MADDPG with AoI-Enhanced/model/marl_model/', 'AoI-Enh', '#9467bd'),
    ('11-Modified MADDPG with ParamShare/model/marl_model/', 'ParamShare', '#8c564b'),
    ('12-Modified MADDPG with AoI-ParamShare/model/marl_model/', 'AoI-ParamShare', '#e6550d'),
]

names = []
cam_rates = []
v2i_rates = []
colors = []

for folder, label, color in algos:
    demand = scipy.io.loadmat(folder + 'demand.mat')['demand']
    v2i = scipy.io.loadmat(folder + 'V2I.mat')['V2I']
    names.append(label)
    cam_rates.append(compute_cam(demand))
    v2i_rates.append(np.mean(v2i[:, -50:, :]))
    colors.append(color)

fig, (ax_cam, ax_v2i) = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(names))

# Left: CAM delivery rate
bars_cam = ax_cam.bar(x, cam_rates, color=colors, edgecolor='black', linewidth=0.5)
ax_cam.set_ylabel('CAM Delivery Rate (%)', fontsize=13)
ax_cam.set_xticks(x)
ax_cam.set_xticklabels(names, fontsize=9, rotation=15, ha='right')
ax_cam.set_ylim(0, 115)
ax_cam.grid(True, alpha=0.2, axis='y')
ax_cam.set_title('CAM Delivery Rate', fontsize=13, fontweight='bold')
for bar, val in zip(bars_cam, cam_rates):
    ax_cam.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Right: V2I rate
bars_v2i = ax_v2i.bar(x, v2i_rates, color=colors, edgecolor='black', linewidth=0.5)
ax_v2i.set_ylabel('V2I Rate', fontsize=13)
ax_v2i.set_xticks(x)
ax_v2i.set_xticklabels(names, fontsize=9, rotation=15, ha='right')
ax_v2i.set_ylim(0, 430)
ax_v2i.grid(True, alpha=0.2, axis='y')
ax_v2i.set_title('V2I Rate', fontsize=13, fontweight='bold')
for bar, val in zip(bars_v2i, v2i_rates):
    ax_v2i.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)

fig.tight_layout()
fig.savefig('figures/fig6_cam_v2i_comparison.png', dpi=300, bbox_inches='tight')
print('Saved figures/fig6_cam_v2i_comparison.png')
