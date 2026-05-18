"""
Plot all figures for AoI-V2X paper reproduction.
Generates Fig. 1-8 matching the paper's results section.

Usage: python plot_results.py
"""

import scipy.io as sio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

REPO = '/archive/hot2/chj/workspace/homework/AoI-V2X-IEEE-TVT-2023'
OUT_DIR = f'{REPO}/figures'
Path(OUT_DIR).mkdir(exist_ok=True)

COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c']
LABELS = [
    'Modified MADDPG with TDec',
    'Modified MADDPG',
    'MADDPG-FDec',
    'DDPG',
]
SMOOTH_WINDOW = 20


def smooth(data, window=SMOOTH_WINDOW):
    """Moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def load_mat(path, key):
    return sio.loadmat(path)[key]


def get_total_reward(alg_dir):
    """Get per-episode total reward averaged over agents for each algorithm."""
    if 'TDec' in alg_dir or 'Attention' in alg_dir and 'AoI' not in alg_dir or 'Attention+AoI' in alg_dir:
        # Has reward_t1 and reward_t2
        try:
            rt1 = load_mat(f'{alg_dir}/model/marl_model/reward_t1.mat', 'reward_t1')
            rt2 = load_mat(f'{alg_dir}/model/marl_model/reward_t2.mat', 'reward_t2')
            total = rt1 + rt2
            return np.mean(total, axis=0)
        except FileNotFoundError:
            pass
    # Has single reward.mat
    r = load_mat(f'{alg_dir}/model/marl_model/reward.mat', 'reward')
    return np.mean(r, axis=0)


def get_aoi(alg_dir):
    """Get per-episode AoI averaged over agents."""
    aoi = load_mat(f'{alg_dir}/model/marl_model/AoI.mat', 'AoI')
    return np.mean(aoi, axis=0)


def get_metric(alg_dir, metric_name):
    """Get last-100-episode average of a metric across all steps and agents."""
    data = load_mat(f'{alg_dir}/model/marl_model/{metric_name}.mat', metric_name)
    # data shape: (n_platoon, 100, n_step) — rolling window of last 100 episodes
    return np.mean(data)


def get_metric_last_episodes(alg_dir, metric_name, last_n=10):
    """Get per-episode average of a metric for the last N episodes."""
    data = load_mat(f'{alg_dir}/model/marl_model/{metric_name}.mat', metric_name)
    return data


# ==========================================================================
# Fig. 1: Task 1 and Task 2 reward convergence for the proposed method
# (Modified MADDPG with TDec only)
# ==========================================================================
def plot_fig1():
    print('Plotting Fig. 1: Task-specific rewards (TDec)')
    alg_dir = f'{REPO}/1-Modified MADDPG with TDec'
    rt1 = load_mat(f'{alg_dir}/model/marl_model/reward_t1.mat', 'reward_t1')
    rt2 = load_mat(f'{alg_dir}/model/marl_model/reward_t2.mat', 'reward_t2')

    avg_t1 = np.mean(rt1, axis=0)
    avg_t2 = np.mean(rt2, axis=0)
    episodes = np.arange(1, len(avg_t1) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Task 1
    ax1.plot(episodes, avg_t1, color=COLORS[0], alpha=0.2, linewidth=0.5)
    s1 = smooth(avg_t1)
    ax1.plot(np.arange(len(s1)) + SMOOTH_WINDOW // 2 + 1, s1, color=COLORS[0], linewidth=2, label='Task 1 (V2V)')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Average Reward', fontsize=12)
    ax1.set_title('Task 1 Reward Convergence', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Task 2
    ax2.plot(episodes, avg_t2, color=COLORS[1], alpha=0.2, linewidth=0.5)
    s2 = smooth(avg_t2)
    ax2.plot(np.arange(len(s2)) + SMOOTH_WINDOW // 2 + 1, s2, color=COLORS[1], linewidth=2, label='Task 2 (V2I+AoI)')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Average Reward', fontsize=12)
    ax2.set_title('Task 2 Reward Convergence', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig1_task_rewards.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved fig1_task_rewards.png')


# ==========================================================================
# Fig. 2: Total reward comparison of all 4 algorithms
# ==========================================================================
def plot_fig2():
    print('Plotting Fig. 2: Total reward comparison')
    alg_dirs = [
        f'{REPO}/1-Modified MADDPG with TDec',
        f'{REPO}/2-Modified MADDPG',
        f'{REPO}/3-MADDPG_FDec',
        f'{REPO}/4-DDPG',
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (alg_dir, label) in enumerate(zip(alg_dirs, LABELS)):
        reward = get_total_reward(alg_dir)
        episodes = np.arange(1, len(reward) + 1)
        ax.plot(episodes, reward, color=COLORS[i], alpha=0.15, linewidth=0.5)
        s = smooth(reward)
        offset = SMOOTH_WINDOW // 2 + 1
        ax.plot(np.arange(len(s)) + offset, s, color=COLORS[i], linewidth=2, label=label)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Total Reward', fontsize=12)
    ax.set_title('Reward Convergence Comparison', fontsize=14)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig2_reward_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved fig2_reward_comparison.png')


# ==========================================================================
# Fig. 3: AoI convergence comparison
# ==========================================================================
def plot_fig3():
    print('Plotting Fig. 3: AoI comparison')
    alg_dirs = [
        f'{REPO}/1-Modified MADDPG with TDec',
        f'{REPO}/2-Modified MADDPG',
        f'{REPO}/3-MADDPG_FDec',
        f'{REPO}/4-DDPG',
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (alg_dir, label) in enumerate(zip(alg_dirs, LABELS)):
        aoi = get_aoi(alg_dir)
        episodes = np.arange(1, len(aoi) + 1)
        ax.plot(episodes, aoi, color=COLORS[i], alpha=0.15, linewidth=0.5)
        s = smooth(aoi)
        offset = SMOOTH_WINDOW // 2 + 1
        ax.plot(np.arange(len(s)) + offset, s, color=COLORS[i], linewidth=2, label=label)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average AoI', fontsize=12)
    ax.set_title('Age of Information Convergence Comparison', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig3_aoi_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved fig3_aoi_comparison.png')


# ==========================================================================
# Fig. 4: V2I rate comparison
# ==========================================================================
def plot_fig4():
    print('Plotting Fig. 4: V2I rate comparison')
    alg_dirs = [
        f'{REPO}/1-Modified MADDPG with TDec',
        f'{REPO}/2-Modified MADDPG',
        f'{REPO}/3-MADDPG_FDec',
        f'{REPO}/4-DDPG',
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (alg_dir, label) in enumerate(zip(alg_dirs, LABELS)):
        v2i = get_metric_last_episodes(alg_dir, 'V2I')
        # Average over agents and episodes, get per-step rate
        avg_per_ep = np.mean(np.mean(v2i, axis=0), axis=0)
        steps = np.arange(1, len(avg_per_ep) + 1)
        ax.plot(steps, avg_per_ep, color=COLORS[i], linewidth=1.5, label=label)

    ax.set_xlabel('Time Step within Episode', fontsize=12)
    ax.set_ylabel('Average V2I Rate (bps)', fontsize=12)
    ax.set_title('V2I Communication Rate (last 100 episodes)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig4_v2i_rate.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved fig4_v2i_rate.png')


# ==========================================================================
# Fig. 5: V2V rate comparison
# ==========================================================================
def plot_fig5():
    print('Plotting Fig. 5: V2V rate comparison')
    alg_dirs = [
        f'{REPO}/1-Modified MADDPG with TDec',
        f'{REPO}/2-Modified MADDPG',
        f'{REPO}/3-MADDPG_FDec',
        f'{REPO}/4-DDPG',
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (alg_dir, label) in enumerate(zip(alg_dirs, LABELS)):
        v2v = get_metric_last_episodes(alg_dir, 'V2V')
        avg_per_ep = np.mean(np.mean(v2v, axis=0), axis=0)
        steps = np.arange(1, len(avg_per_ep) + 1)
        ax.plot(steps, avg_per_ep, color=COLORS[i], linewidth=1.5, label=label)

    ax.set_xlabel('Time Step within Episode', fontsize=12)
    ax.set_ylabel('Average V2V Rate (bps)', fontsize=12)
    ax.set_title('V2V Communication Rate (last 100 episodes)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig5_v2v_rate.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved fig5_v2v_rate.png')


# ==========================================================================
# Fig. 6: Power allocation comparison
# ==========================================================================
def plot_fig6():
    print('Plotting Fig. 6: Power allocation')
    alg_dirs = [
        f'{REPO}/1-Modified MADDPG with TDec',
        f'{REPO}/2-Modified MADDPG',
        f'{REPO}/3-MADDPG_FDec',
        f'{REPO}/4-DDPG',
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (alg_dir, label) in enumerate(zip(alg_dirs, LABELS)):
        power = get_metric_last_episodes(alg_dir, 'power')
        avg_per_ep = np.mean(np.mean(power, axis=0), axis=0)
        steps = np.arange(1, len(avg_per_ep) + 1)
        ax.plot(steps, avg_per_ep, color=COLORS[i], linewidth=1.5, label=label)

    ax.set_xlabel('Time Step within Episode', fontsize=12)
    ax.set_ylabel('Average Power (dBm)', fontsize=12)
    ax.set_title('Power Allocation (last 100 episodes)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig6_power.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved fig6_power.png')


# ==========================================================================
# Fig. 7: V2V Demand remaining comparison
# ==========================================================================
def plot_fig7():
    print('Plotting Fig. 7: V2V Demand')
    alg_dirs = [
        f'{REPO}/1-Modified MADDPG with TDec',
        f'{REPO}/2-Modified MADDPG',
        f'{REPO}/3-MADDPG_FDec',
        f'{REPO}/4-DDPG',
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (alg_dir, label) in enumerate(zip(alg_dirs, LABELS)):
        demand = get_metric_last_episodes(alg_dir, 'demand')
        avg_per_ep = np.mean(np.mean(demand, axis=0), axis=0)
        steps = np.arange(1, len(avg_per_ep) + 1)
        ax.plot(steps, avg_per_ep, color=COLORS[i], linewidth=1.5, label=label)

    ax.set_xlabel('Time Step within Episode', fontsize=12)
    ax.set_ylabel('Average V2V Demand Remaining (bits)', fontsize=12)
    ax.set_title('V2V Demand Remaining (last 100 episodes)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig7_demand.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved fig7_demand.png')


# ==========================================================================
# Fig. 8: AoI evolution within an episode
# ==========================================================================
def plot_fig8():
    print('Plotting Fig. 8: AoI evolution within episode')
    alg_dirs = [
        f'{REPO}/1-Modified MADDPG with TDec',
        f'{REPO}/2-Modified MADDPG',
        f'{REPO}/3-MADDPG_FDec',
        f'{REPO}/4-DDPG',
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (alg_dir, label) in enumerate(zip(alg_dirs, LABELS)):
        aoi_evo = get_metric_last_episodes(alg_dir, 'AoI_evolution')
        avg_per_ep = np.mean(np.mean(aoi_evo, axis=0), axis=0)
        steps = np.arange(1, len(avg_per_ep) + 1)
        ax.plot(steps, avg_per_ep, color=COLORS[i], linewidth=1.5, label=label)

    ax.set_xlabel('Time Step within Episode', fontsize=12)
    ax.set_ylabel('Average AoI', fontsize=12)
    ax.set_title('AoI Evolution within Episode (last 100 episodes)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig8_aoi_evolution.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved fig8_aoi_evolution.png')


# ==========================================================================
# Summary table
# ==========================================================================
def print_summary():
    print('\n' + '=' * 70)
    print('SUMMARY: Final results comparison (last 10 episodes average)')
    print('=' * 70)
    alg_dirs = [
        f'{REPO}/1-Modified MADDPG with TDec',
        f'{REPO}/2-Modified MADDPG',
        f'{REPO}/3-MADDPG_FDec',
        f'{REPO}/4-DDPG',
    ]
    print(f'{"Algorithm":<35s} {"AoI":>8s} {"Reward":>10s} {"V2I":>8s} {"V2V":>8s} {"Power":>8s}')
    print('-' * 70)
    for i, (alg_dir, label) in enumerate(zip(alg_dirs, LABELS)):
        aoi = np.mean(get_aoi(alg_dir)[-10:])
        reward = np.mean(get_total_reward(alg_dir)[-10:])
        v2i_data = load_mat(f'{alg_dir}/model/marl_model/V2I.mat', 'V2I')
        v2i = np.mean(v2i_data[:, -10:, :])
        v2v_data = load_mat(f'{alg_dir}/model/marl_model/V2V.mat', 'V2V')
        v2v = np.mean(v2v_data[:, -10:, :])
        power_data = load_mat(f'{alg_dir}/model/marl_model/power.mat', 'power')
        power = np.mean(power_data[:, -10:, :])
        print(f'{label:<35s} {aoi:>8.2f} {reward:>10.4f} {v2i:>8.1f} {v2v:>8.1f} {power:>8.2f}')
    print('=' * 70)


if __name__ == '__main__':
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    plot_fig5()
    plot_fig6()
    plot_fig7()
    plot_fig8()
    print_summary()
    print(f'\nAll figures saved to {OUT_DIR}/')
