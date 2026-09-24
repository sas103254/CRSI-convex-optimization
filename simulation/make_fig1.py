import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path

fig, ax = plt.subplots(figsize=(7, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 13)
ax.axis('off')

def box(x, y, w, h, text, color, fontsize=8.3):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                        linewidth=1.1, edgecolor='black', facecolor=color)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize, wrap=True)

def arrow(x1, y1, x2, y2, style='-|>', color='black', connectionstyle=None, lw=1.2, ls='-'):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                         color=color, lw=lw, linestyle=ls,
                         connectionstyle=connectionstyle)
    ax.add_patch(a)

# Phase 1
ax.text(0.15, 12.55, 'Phase 1: Real-Time System State', fontsize=9.5, fontweight='bold')
box(0.3, 11.2, 4.2, 1.15, 'Incoming Workloads\nTask demands $r_i$, latency\nmatrix $d_{ij}$', '#dbe9f6')
box(5.0, 11.2, 4.5, 1.15, 'CRSI State Assessor\n$H_j,R_j,P_j,S_j \\rightarrow$ CRSI$_j$\n(thermal, util., link, security)', '#dbe9f6')
box(2.6, 9.8, 4.6, 1.0, 'State History Buffer\n$X^{(t-1)}$, server capacities $C_j$', '#dbe9f6')
arrow(2.4, 11.2, 4.6, 10.5, connectionstyle='arc3,rad=-0.1')
arrow(7.25, 11.2, 5.2, 10.5, connectionstyle='arc3,rad=0.1')

# Phase 2
ax.text(0.15, 9.35, 'Phase 2: Adaptive Parameter Regulation', fontsize=9.5, fontweight='bold')
box(1.4, 8.0, 7.2, 1.05,
    'Adaptive Weight Controller\nz-score signals -> softmax -> EMA blend\n'
    '$w_{t}=(1-\\eta)w_{t-1}+\\eta\\,\\mathrm{softmax}(z/\\tau)$\nupdates $(\\alpha,\\beta,\\gamma,\\delta)$, $\\sum=1$',
    '#fde3cf')
arrow(4.9, 9.8, 4.9, 9.05)

# Phase 3
ax.text(0.15, 7.55, 'Phase 3: Convex QP Solver Core', fontsize=9.5, fontweight='bold')
box(0.3, 5.55, 4.3, 1.85,
    'Strictly Convex Objective\n$\\min_X \\sum_j[\\alpha u_j^2+\\beta E_j+\\gamma L_j$\n'
    '$+\\delta(1-\\mathrm{CRSI}_j)u_j]+\\lambda\\|X-X^{(t-1)}\\|_F^2$',
    '#e3f0d8')
box(5.0, 5.55, 4.5, 1.85,
    'Linear Constraints\nTask completeness $\\sum_j x_{ij}=1$\nCapacity $u_j \\leq C_j$\nBox domain $0\\leq x_{ij}\\leq1$',
    '#e3f0d8')
box(1.9, 4.15, 6.2, 1.0, 'Interior-Point Method (IPM)\nPolynomial-time convergence to $X^*$', '#e3f0d8')
arrow(4.9, 8.0, 2.6, 7.4, connectionstyle='arc3,rad=-0.1')
arrow(4.9, 8.0, 7.2, 7.4, connectionstyle='arc3,rad=0.1')
arrow(2.6, 5.55, 4.5, 5.15)
arrow(7.2, 5.55, 5.5, 5.15)

# Phase 4
ax.text(0.15, 3.75, 'Phase 4: Discretization, Migration, Deployment', fontsize=9.5, fontweight='bold')
box(0.6, 2.35, 8.0, 1.05,
    'CRSI-Weighted Max-Argument Rounding\n'
    'Converts continuous $X^*$ to discrete $\\{0,1\\}$ assignment,\n'
    'validated against capacity feasibility (Sec. VI-E)',
    '#f3d9e6')
box(0.6, 0.9, 8.0, 1.05, 'Cloud Workload Orchestrator\nDeploys containers/VMs, executes controlled migrations across servers $1,\\ldots,M$', '#f3d9e6')
arrow(4.9, 4.15, 4.6, 3.4)
arrow(4.6, 2.35, 4.6, 1.95)

# Feedback loop
arrow(0.6, 1.4, 0.15, 1.4, style='-', color='#555555')
ax.plot([0.15, 0.15], [1.4, 12.05], color='#555555', linewidth=1.2, linestyle='--')
arrow(0.15, 12.05, 0.3, 11.85, color='#555555', ls='--')
ax.text(-0.05, 6.7, 'Feedback: deployed $X$ becomes $X^{(t-1)}$; controller signals\n'
                     '(load imbalance, energy, latency, worst-case stress) feed $t+1$',
        rotation=90, ha='center', va='center', fontsize=7.3, color='#555555')

plt.tight_layout()
plt.savefig('../results/figs/fig1_architecture.png', dpi=160, bbox_inches='tight')
print("done")
