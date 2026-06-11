"""Fig 6: Three separate bar charts — AoI, CAM delivery rate, Training time."""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12

BASE = '/archive/hot2/chj/workspace/homework/AoI-V2X-IEEE-TVT-2023/'

algos = [
    (BASE + '2-Modified MADDPG/model/marl_model/', 'MADDPG', '#ff7f0e'),
    (BASE + '3-MADDPG_FDec/model/marl_model/', 'FDec', '#2ca02c'),
    (BASE + '1-Modified MADDPG with TDec/model/marl_model/', 'TDec', '#1f77b4'),
    (BASE + '6-Modified MADDPG with AoI-Enhanced/model/marl_model/', 'AoI-Enh', '#9467bd'),
    (BASE + '11-Modified MADDPG with ParamShare/model/marl_model/', 'ParamShare', '#8c564b'),
    (BASE + '12-Modified MADDPG with AoI-ParamShare/model/marl_model/', 'AoI-Param', '#e6550d'),
]

# Training time (min) — from table_comparison.txt, not stored in .mat
time_values = [73.2, 79.8, 88.7, 76.8, 61.2, 68.6]


def compute_cam(demand):
    last50 = demand[:, -50:, :]
    return np.mean(np.sum(last50[:, :, -1] <= 0, axis=0) == 5) * 100


def compute_aoi(aoi):
    return np.mean(aoi[:, -50:])


names = []
colors = []
aoi_vals = []
cam_vals = []

for folder, label, color in algos:
    demand = scipy.io.loadmat(folder + 'demand.mat')['demand']
    aoi = scipy.io.loadmat(folder + 'AoI.mat')['AoI']
    names.append(label)
    colors.append(color)
    aoi_vals.append(compute_aoi(aoi))
    cam_vals.append(compute_cam(demand))

x = np.arange(len(names))
DIVIDER = 2.5  # between TDec (idx=2) and AoI-Enh (idx=3)


def add_value_labels(ax, bars, fmt, offset):
    for bar, val in zip(bars, fmt):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')


def add_divider(ax):
    ax.axvline(x=DIVIDER, color='gray', linestyle='--', linewidth=1.2)


# ---------- Fig 6a: AoI ----------
fig_a, ax_a = plt.subplots(figsize=(8, 5))
bars_a = ax_a.bar(x, aoi_vals, color=colors, edgecolor='black', linewidth=0.5)
ax_a.set_ylabel('Age of Information (AoI)', fontsize=13)
ax_a.set_xticks(x)
ax_a.set_xticklabels(names, fontsize=9, rotation=15, ha='right')
ax_a.grid(True, alpha=0.2, axis='y')
ax_a.set_title('AoI Comparison (Lower is Better)', fontsize=13, fontweight='bold')
add_value_labels(ax_a, bars_a, [f'{v:.2f}' for v in aoi_vals], offset=0.5)
add_divider(ax_a)
fig_a.tight_layout()
fig_a.savefig(BASE + 'figures/fig6a_aoi_comparison.png', dpi=300, bbox_inches='tight')
print('Saved figures/fig6a_aoi_comparison.png')
plt.close(fig_a)

# ---------- Fig 6b: CAM delivery rate ----------
fig_c, ax_c = plt.subplots(figsize=(8, 5))
bars_c = ax_c.bar(x, cam_vals, color=colors, edgecolor='black', linewidth=0.5)
ax_c.set_ylabel('CAM Delivery Rate (%)', fontsize=13)
ax_c.set_xticks(x)
ax_c.set_xticklabels(names, fontsize=9, rotation=15, ha='right')
ax_c.set_ylim(0, 115)
ax_c.grid(True, alpha=0.2, axis='y')
ax_c.set_title('CAM Delivery Rate (Higher is Better)', fontsize=13, fontweight='bold')
add_value_labels(ax_c, bars_c, [f'{v:.0f}%' for v in cam_vals], offset=1)
add_divider(ax_c)
fig_c.tight_layout()
fig_c.savefig(BASE + 'figures/fig6b_cam_comparison.png', dpi=300, bbox_inches='tight')
print('Saved figures/fig6b_cam_comparison.png')
plt.close(fig_c)

# ---------- Fig 6c: Training time ----------
fig_t, ax_t = plt.subplots(figsize=(8, 5))
bars_t = ax_t.bar(x, time_values, color=colors, edgecolor='black', linewidth=0.5)
ax_t.set_ylabel('Training Time (min)', fontsize=13)
ax_t.set_xticks(x)
ax_t.set_xticklabels(names, fontsize=9, rotation=15, ha='right')
ax_t.grid(True, alpha=0.2, axis='y')
ax_t.set_title('Training Time (Lower is Better)', fontsize=13, fontweight='bold')
add_value_labels(ax_t, bars_t, [f'{v:.1f}' for v in time_values], offset=1)
add_divider(ax_t)
fig_t.tight_layout()
fig_t.savefig(BASE + 'figures/fig6c_time_comparison.png', dpi=300, bbox_inches='tight')
print('Saved figures/fig6c_time_comparison.png')
plt.close(fig_t)
