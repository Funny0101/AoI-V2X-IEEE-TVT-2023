"""
Plot comparison: baseline TDec vs 3 new algorithms (5, 6, 7).
"""

import scipy.io as sio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = '/archive/hot2/chj/workspace/homework/AoI-V2X-IEEE-TVT-2023'
OUT_DIR = f'{REPO}/figures'

COLORS = ['#2c3e50', '#e74c3c', '#3498db', '#2ecc71']
LABELS = [
    'TDec (Baseline)',
    'Attention',
    'AoI-Enhanced',
    'Attention+AoI',
]
ALGS = [
    '1-Modified MADDPG with TDec',
    '5-Modified MADDPG with Attention',
    '6-Modified MADDPG with AoI-Enhanced',
    '7-Modified MADDPG with Attention+AoI',
]

W = 15

def smooth(data, w=W):
    return np.convolve(data, np.ones(w)/w, mode='valid')


def get_total_reward(alg_dir):
    rt1 = sio.loadmat(f'{alg_dir}/model/marl_model/reward_t1.mat')['reward_t1']
    rt2 = sio.loadmat(f'{alg_dir}/model/marl_model/reward_t2.mat')['reward_t2']
    return np.mean(rt1 + rt2, axis=0)


def get_aoi(alg_dir):
    aoi = sio.loadmat(f'{alg_dir}/model/marl_model/AoI.mat')['AoI']
    return np.mean(aoi, axis=0)


fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- Reward Convergence ---
ax = axes[0, 0]
for i, (alg, label) in enumerate(zip(ALGS, LABELS)):
    r = get_total_reward(f'{REPO}/{alg}')
    ep = np.arange(1, len(r)+1)
    ax.plot(ep, r, color=COLORS[i], alpha=0.12, linewidth=0.5)
    s = smooth(r)
    ax.plot(np.arange(len(s)) + W//2+1, s, color=COLORS[i], linewidth=2, label=label)
ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('Average Total Reward', fontsize=12)
ax.set_title('(a) Reward Convergence', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- AoI Convergence ---
ax = axes[0, 1]
for i, (alg, label) in enumerate(zip(ALGS, LABELS)):
    aoi = get_aoi(f'{REPO}/{alg}')
    ep = np.arange(1, len(aoi)+1)
    ax.plot(ep, aoi, color=COLORS[i], alpha=0.12, linewidth=0.5)
    s = smooth(aoi)
    ax.plot(np.arange(len(s)) + W//2+1, s, color=COLORS[i], linewidth=2, label=label)
ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('Average AoI', fontsize=12)
ax.set_title('(b) AoI Convergence', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- V2I Rate ---
ax = axes[1, 0]
for i, (alg, label) in enumerate(zip(ALGS, LABELS)):
    v2i = sio.loadmat(f'{REPO}/{alg}/model/marl_model/V2I.mat')['V2I']
    avg = np.mean(np.mean(v2i, axis=0), axis=0)
    steps = np.arange(1, len(avg)+1)
    ax.plot(steps, avg, color=COLORS[i], linewidth=1.5, label=label)
ax.set_xlabel('Time Step', fontsize=12)
ax.set_ylabel('V2I Rate (bps)', fontsize=12)
ax.set_title('(c) V2I Rate (last 100 episodes)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- V2V Rate ---
ax = axes[1, 1]
for i, (alg, label) in enumerate(zip(ALGS, LABELS)):
    v2v = sio.loadmat(f'{REPO}/{alg}/model/marl_model/V2V.mat')['V2V']
    avg = np.mean(np.mean(v2v, axis=0), axis=0)
    steps = np.arange(1, len(avg)+1)
    ax.plot(steps, avg, color=COLORS[i], linewidth=1.5, label=label)
ax.set_xlabel('Time Step', fontsize=12)
ax.set_ylabel('V2V Rate (bps)', fontsize=12)
ax.set_title('(d) V2V Rate (last 100 episodes)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle('Baseline (TDec) vs Proposed Improvements', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_comparison_new.png', dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved fig_comparison_new.png')


# --- Bar chart summary ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

short_labels = ['TDec\n(Baseline)', 'Attn\n(5)', 'AoI-Enh\n(6)', 'Attn+AoI\n(7)']
metrics = {}
for i, alg in enumerate(ALGS):
    aoi = np.mean(get_aoi(f'{REPO}/{alg}')[-10:])
    rwd = np.mean(get_total_reward(f'{REPO}/{alg}')[-10:])
    pwr = np.mean(sio.loadmat(f'{REPO}/{alg}/model/marl_model/power.mat')['power'][:,-10:,:])
    metrics[i] = {'aoi': aoi, 'rwd': rwd, 'pwr': pwr}

# AoI bar
ax = axes[0]
vals = [metrics[i]['aoi'] for i in range(4)]
bars = ax.bar(short_labels, vals, color=COLORS, alpha=0.85, edgecolor='white', linewidth=1.5)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Average AoI', fontsize=12)
ax.set_title('Average AoI (lower is better)', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Reward bar
ax = axes[1]
vals = [metrics[i]['rwd'] for i in range(4)]
bars = ax.bar(short_labels, vals, color=COLORS, alpha=0.85, edgecolor='white', linewidth=1.5)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Total Reward', fontsize=12)
ax.set_title('Total Reward (higher is better)', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Power bar
ax = axes[2]
vals = [metrics[i]['pwr'] for i in range(4)]
bars = ax.bar(short_labels, vals, color=COLORS, alpha=0.85, edgecolor='white', linewidth=1.5)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Power (dBm)', fontsize=12)
ax.set_title('Power (lower is better)', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_bar_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved fig_bar_comparison.png')
