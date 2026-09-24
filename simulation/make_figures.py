import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

OUT = '../results/figs'

with open('../results/exp1_results.json') as f:
    res = json.load(f)
with open('../results/exp1_summary.json') as f:
    summ = json.load(f)
with open('../results/exp2_rounding_gap.json') as f:
    gapres = json.load(f)
with open('../results/exp3_lambda_sweep.json') as f:
    lam_sweep = json.load(f)
with open('../results/exp4_scalability.json') as f:
    scal = json.load(f)
weight_log = np.load('../results/weight_log.npy')

T = len(res['proposed_fixed']['u_deg'])
t_axis = np.arange(T)
incident_step = 5

# ---------------- Fig 2: load shed from degraded server ----------------
plt.figure(figsize=(6, 4))
plt.plot(t_axis, res['proposed_fixed']['u_deg'], 'o-', label='Proposed (fixed weights)', color='#1f77b4')
plt.plot(t_axis, res['proposed_adaptive']['u_deg'], 's-', label='Proposed (adaptive weights)', color='#2ca02c')
plt.plot(t_axis, res['round_robin']['u_deg'], '^-', label='Round-Robin', color='#d62728')
plt.plot(t_axis, res['greedy']['u_deg'], 'D-', label='Greedy Least-Loaded', color='#ff7f0e')
plt.plot(t_axis, res['crsi_threshold']['u_deg'], 'v-', label='CRSI-Threshold Heuristic', color='#9467bd')
plt.axvline(incident_step - 0.5, color='gray', linestyle=':', label='stability incident')
plt.xlabel('Time step $t$')
plt.ylabel('Normalized load on degraded server $u_j/C_j$')
plt.title('Load Shed from Degraded Server')
plt.legend(fontsize=7, loc='upper right')
plt.tight_layout()
plt.savefig(f'{OUT}/fig2_load_shed.png')
plt.close()

# ---------------- Fig 3: migration volume, proposed vs no-reg ----------------
plt.figure(figsize=(6, 4))
plt.plot(t_axis, res['proposed_fixed']['migration'], 'o-', label='Proposed (CRSI + reg.)', color='#1f77b4')
plt.plot(t_axis, res['no_reg']['migration'], 's--', label='Proposed (no reg., $\\lambda=0$)', color='#d62728')
plt.xlabel('Time step $t$')
plt.ylabel('Migration volume $\\|X_t - X_{t-1}\\|_F$')
plt.title('Effect of Regularization on Task Migration')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/fig3_migration.png')
plt.close()

# ---------------- Fig 4: CRSI-weighted stress exposure ----------------
plt.figure(figsize=(6, 4))
plt.plot(t_axis, res['proposed_fixed']['stress_exposure'], 'o-', label='Proposed (fixed)', color='#1f77b4')
plt.plot(t_axis, res['proposed_adaptive']['stress_exposure'], 's-', label='Proposed (adaptive)', color='#2ca02c')
plt.plot(t_axis, res['round_robin']['stress_exposure'], '^-', label='Round-Robin', color='#d62728')
plt.plot(t_axis, res['greedy']['stress_exposure'], 'D-', label='Greedy', color='#ff7f0e')
plt.plot(t_axis, res['crsi_threshold']['stress_exposure'], 'v-', label='CRSI-Threshold', color='#9467bd')
plt.axvline(incident_step - 0.5, color='gray', linestyle=':')
plt.xlabel('Time step $t$')
plt.ylabel('$\\sum_j (1-\\mathrm{CRSI}_j) u_j$')
plt.title('CRSI-Weighted Stress Exposure')
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_stress_exposure.png')
plt.close()

# ---------------- Fig 5: load-balancing quality over time ----------------
plt.figure(figsize=(6, 4))
plt.plot(t_axis, res['proposed_fixed']['load_std'], 'o-', label='Proposed (fixed)', color='#1f77b4')
plt.plot(t_axis, res['round_robin']['load_std'], '^-', label='Round-Robin', color='#d62728')
plt.plot(t_axis, res['greedy']['load_std'], 'D-', label='Greedy Least-Loaded', color='#ff7f0e')
plt.plot(t_axis, res['crsi_threshold']['load_std'], 'v-', label='CRSI-Threshold', color='#9467bd')
plt.xlabel('Time step $t$')
plt.ylabel('Std. dev. of server utilization $u_j$')
plt.title('Load-Balancing Quality Over Time')
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_load_balance.png')
plt.close()

# ---------------- Fig 6 (NEW): adaptive weight trajectory ----------------
plt.figure(figsize=(6, 4))
labels = ['$\\alpha$ (load)', '$\\beta$ (energy)', '$\\gamma$ (latency)', '$\\delta$ (stability)']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
tw = np.arange(len(weight_log))
for k in range(4):
    plt.plot(tw, weight_log[:, k], 'o-', label=labels[k], color=colors[k])
plt.axvline(incident_step - 0.5, color='gray', linestyle=':', label='stability incident')
plt.xlabel('Time step $t$')
plt.ylabel('Weight value')
plt.title('Adaptive Weight Controller Trajectory')
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(f'{OUT}/fig6_adaptive_weights.png')
plt.close()

# ---------------- Fig 7 (NEW): lambda sensitivity trade-off ----------------
keys_sorted = sorted(lam_sweep.keys(), key=lambda x: float(x))
lam_vals = [float(k) for k in keys_sorted]
mig_vals = [lam_sweep[k]['mean_migration'] for k in keys_sorted]
lat_vals = [lam_sweep[k]['mean_latency_norm'] for k in keys_sorted]

fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(lam_vals, mig_vals, 'o-', color='#1f77b4', label='Mean migration volume')
ax1.set_xlabel('Regularization weight $\\lambda$')
ax1.set_ylabel('Mean migration volume', color='#1f77b4')
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax2 = ax1.twinx()
ax2.plot(lam_vals, lat_vals, 's--', color='#d62728', label='Mean normalized latency')
ax2.set_ylabel('Mean normalized latency', color='#d62728')
ax2.tick_params(axis='y', labelcolor='#d62728')
plt.title('$\\lambda$ Sensitivity: Migration vs. Latency Trade-off')
fig.tight_layout()
plt.savefig(f'{OUT}/fig7_lambda_sensitivity.png')
plt.close()

# ---------------- Fig 8 (NEW): solver runtime scalability ----------------
Ms = [r['vars'] for r in scal]
ts = [r['mean_solve_time_s'] for r in scal]
plt.figure(figsize=(6, 4))
plt.loglog(Ms, ts, 'o-', color='#1f77b4')
for r in scal:
    plt.annotate(f"M={r['M']}\nN={r['N']}", (r['vars'], r['mean_solve_time_s']),
                 textcoords="offset points", xytext=(5, 5), fontsize=7)
plt.xlabel('Number of decision variables $N \\times M$ (log scale)')
plt.ylabel('Mean solve time (s, log scale)')
plt.title('Interior-Point Solver Runtime vs. Problem Size')
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/fig8_scalability.png')
plt.close()

# ---------------- Fig 9 (NEW): rounding optimality gap ----------------
plt.figure(figsize=(6, 4))
plt.bar(range(len(gapres['gap_pct_per_step'])), gapres['gap_pct_per_step'], color='#1f77b4')
plt.axhline(gapres['mean_gap_pct'], color='#d62728', linestyle='--',
            label=f"mean = {gapres['mean_gap_pct']:.2f}%")
plt.xlabel('Time step $t$')
plt.ylabel('Rounding optimality gap (%)')
plt.title('CRSI-Weighted Rounding: Discretization Optimality Gap')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/fig9_rounding_gap.png')
plt.close()

print("All figures generated.")
