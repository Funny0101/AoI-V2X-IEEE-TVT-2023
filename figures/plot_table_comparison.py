"""Quantitative comparison table with training time and CAM delivery. Saves to figures/table_comparison.txt"""

import numpy as np
import scipy.io

def smooth(data, w=20):
    out = np.copy(data)
    for i in range(len(data)):
        start = max(0, i - w + 1)
        out[i] = np.mean(data[start:i+1])
    return out

def load_total_reward(folder, rtype):
    if rtype == 'split':
        t1 = scipy.io.loadmat(folder + 'reward_t1.mat')['reward_t1']
        t2 = scipy.io.loadmat(folder + 'reward_t2.mat')['reward_t2']
        return t1 + t2
    else:
        d = scipy.io.loadmat(folder + 'reward.mat')
        key = [k for k in d.keys() if not k.startswith('_')][0]
        return d[key]

def compute_cam(demand):
    """Compute CAM delivery rate (fraction of episodes where all platoons deliver)."""
    last50 = demand[:, -50:, :]
    all5 = np.mean(np.sum(last50[:,:,-1] <= 0, axis=0) == 5) * 100
    return all5

algos = [
    ('1-Modified MADDPG with TDec/model/marl_model/',             'split',   'TDec MADDPG',    5320.9),
    ('2-Modified MADDPG/model/marl_model/',                        'single',  'MADDPG',         4391.8),
    ('3-MADDPG_FDec/model/marl_model/',                            'single',  'FDec MADDPG',    4786.1),
    ('4-DDPG/model/marl_model/',                                   'single',  'DDPG',            866.0),
    ('6-Modified MADDPG with AoI-Enhanced/model/marl_model/',      'split',   'AoI-Enhanced',   4606.7),
    ('11-Modified MADDPG with ParamShare/model/marl_model/',       'split',   'ParamShare',     3671.0),
    ('12-Modified MADDPG with AoI-ParamShare/model/marl_model/',   'split',   'AoI-ParamShare', 4116.2),
]

lines = []
lines.append("=" * 130)
lines.append(f"{'Algorithm':<18} {'AoI':>8} {'Reward':>10} {'V2I Rate':>10} {'V2V Rate':>10} {'CAM':>6} {'Conv.Ep':>8} {'Time(min)':>10}")
lines.append("=" * 130)

results = []
for folder, rtype, label, train_time in algos:
    aoi = scipy.io.loadmat(folder + 'AoI.mat')['AoI']
    reward = load_total_reward(folder, rtype)
    v2i = scipy.io.loadmat(folder + 'V2I.mat')['V2I']
    v2v = scipy.io.loadmat(folder + 'V2V.mat')['V2V']
    demand = scipy.io.loadmat(folder + 'demand.mat')['demand']

    mean_aoi = np.mean(aoi, axis=0)
    mean_reward = np.mean(reward, axis=0)

    aoi_final = np.mean(mean_aoi[-50:])
    reward_final = np.mean(mean_reward[-50:])
    v2i_final = np.mean(v2i[:, -50:, :])
    v2v_final = np.mean(v2v[:, -50:, :])
    cam = compute_cam(demand)

    s = smooth(mean_aoi, 20)
    target = aoi_final * 1.1
    conv_ep = int(np.argmax(s <= target)) if np.any(s <= target) else -1

    results.append({
        'label': label, 'aoi': aoi_final, 'reward': reward_final,
        'v2i': v2i_final, 'v2v': v2v_final, 'cam': cam,
        'conv_ep': conv_ep, 'time': train_time,
    })

    lines.append(f"{label:<18} {aoi_final:>8.2f} {reward_final:>10.2f} {v2i_final:>10.1f} {v2v_final:>10.1f} {cam:>5.0f}% {conv_ep:>8d} {train_time/60:>10.1f}")

lines.append("=" * 130)

# Changes vs TDec
baseline = results[0]
lines.append("")
lines.append(f"{'Algorithm':<18} {'AoI d%':>8} {'Reward d%':>10} {'V2I d%':>8} {'V2V d%':>8} {'CAM':>6} {'Conv. Speedup':>14} {'Time Saved':>12}")
lines.append("-" * 90)
for r in results:
    aoi_pct = (r['aoi'] - baseline['aoi']) / baseline['aoi'] * 100
    rew_pct = (r['reward'] - baseline['reward']) / abs(baseline['reward']) * 100 if baseline['reward'] != 0 else 0
    v2i_pct = (r['v2i'] - baseline['v2i']) / baseline['v2i'] * 100
    v2v_pct = (r['v2v'] - baseline['v2v']) / baseline['v2v'] * 100
    speedup = f"{baseline['conv_ep']/r['conv_ep']:.2f}x" if r['conv_ep'] > 0 else "N/A"
    time_saved = f"{(1 - r['time']/baseline['time'])*100:+.1f}%"
    lines.append(f"{r['label']:<18} {aoi_pct:>+7.1f}% {rew_pct:>+9.1f}% {v2i_pct:>+7.1f}% {v2v_pct:>+7.1f}% {r['cam']:>5.0f}% {speedup:>14} {time_saved:>12}")

# V2V breakdown section
lines.append("")
lines.append("=== V2V Rate Breakdown (last 50 episodes) ===")
lines.append(f"{'Algorithm':<18} {'V2V (demand>0)':>16} {'V2V (demand=0)':>16} {'CAM 5/5':>8} {'1st delivery':>13}")
lines.append("-" * 75)
for folder, rtype, label, train_time in algos:
    demand = scipy.io.loadmat(folder + 'demand.mat')['demand']
    v2v = scipy.io.loadmat(folder + 'V2V.mat')['V2V']
    last50_d = demand[:, -50:, :]
    last50_v = v2v[:, -50:, :]

    v2v_demand = np.mean(last50_v[last50_d > 0]) if np.any(last50_d > 0) else 0
    v2v_nodemand = np.mean(last50_v[last50_d <= 0]) if np.any(last50_d <= 0) else 0
    cam = compute_cam(demand)

    firsts = []
    for p in range(5):
        for ep in range(50):
            done_steps = np.where(last50_d[p, ep, :] <= 0)[0]
            if len(done_steps) > 0:
                firsts.append(done_steps[0])
    avg_first = np.mean(firsts) if firsts else -1

    lines.append(f"{label:<18} {v2v_demand:>16.0f} {v2v_nodemand:>16.0f} {cam:>7.0f}% {avg_first:>10.1f}/100")

lines.append("")
lines.append("Note: V2V(demand=0) is throughput AFTER CAM delivery. Lower values mean agent")
lines.append("switched to V2I mode to minimize AoI — this is efficient, not a failure.")

# LaTeX
lines.append("")
lines.append("% --- LaTeX Table ---")
lines.append(r"\begin{table}[h]")
lines.append(r"\centering")
lines.append(r"\caption{Quantitative Comparison of Algorithms (Last 50 Episodes Average)}")
lines.append(r"\label{tab:comparison}")
lines.append(r"\begin{tabular}{lcccccccc}")
lines.append(r"\toprule")
lines.append(r"Algorithm & AoI $\downarrow$ & Reward $\uparrow$ & V2I Rate $\uparrow$ & V2V Rate & CAM Delivery $\uparrow$ & Conv. Ep.\ $\downarrow$ & Time (min) $\downarrow$ \\")
lines.append(r"\midrule")
for r in results:
    lines.append(f"  {r['label']} & {r['aoi']:.2f} & {r['reward']:.2f} & {r['v2i']:.1f} & {r['v2v']:.1f} & {r['cam']:.0f}\\% & {r['conv_ep']} & {r['time']/60:.1f} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

output = "\n".join(lines)
print(output)

with open("figures/table_comparison.txt", "w") as f:
    f.write(output)
print("\nSaved figures/table_comparison.txt")
