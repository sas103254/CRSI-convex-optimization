import os
import numpy as np, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (generate_crsi_components, crsi_from_components, solve_qp,
                   crsi_weighted_round, round_robin_assign, greedy_least_loaded_assign,
                   crsi_threshold_heuristic)

rng = np.random.default_rng(99)
M, N, T = 24, 200, 8
incident_server = 3
incident_step = 4
C = rng.uniform(80, 120, M)
e_energy = rng.uniform(0.008, 0.018, M)
alpha, beta, gamma, delta, lam = 0.30, 0.20, 0.20, 0.30, 0.15

methods = ['proposed', 'round_robin', 'greedy', 'crsi_threshold']
state = {m: {'X_prev': np.ones((N, M)) / M, 'rr_ptr': 0} for m in methods}
results = {m: {'u_deg': [], 'migration': [], 'latency_norm': [], 'load_std': [],
               'stress_exposure': []} for m in methods}
prev_crsi_components = None

for t in range(T):
    r = rng.uniform(1, 10, N)
    d_latency = rng.uniform(1, 20, (N, M)) + rng.normal(0, 1.0, (N, M))
    d_latency = np.clip(d_latency, 0.5, None)
    d_lat_norm = d_latency / d_latency.mean()
    incident_active = (t >= incident_step)
    H, R, P, S = generate_crsi_components(M, rng, prev_state=prev_crsi_components,
                                           incident_server=incident_server,
                                           incident_active=incident_active)
    prev_crsi_components = {'H': H, 'R': R, 'P': P, 'S': S}
    CRSI = crsi_from_components(H, R, P, S, (0.25, 0.25, 0.25, 0.25))

    Xc, st, _ = solve_qp(N, M, r, C, d_lat_norm, e_energy, CRSI, state['proposed']['X_prev'],
                          alpha, beta, gamma, delta, lam)
    Xd = crsi_weighted_round(Xc, CRSI)
    u = Xd.T @ r
    results['proposed']['u_deg'].append(u[incident_server] / C[incident_server])
    results['proposed']['migration'].append(np.linalg.norm(Xd - state['proposed']['X_prev'], 'fro'))
    results['proposed']['latency_norm'].append(np.sum(Xd * d_lat_norm) / N)
    results['proposed']['load_std'].append(np.std(u / C))
    results['proposed']['stress_exposure'].append(np.sum((1 - CRSI) * u))
    state['proposed']['X_prev'] = Xd

    Xrr, ptr = round_robin_assign(N, M, r, C, None, state['round_robin']['rr_ptr'])
    state['round_robin']['rr_ptr'] = ptr
    urr = Xrr.T @ r
    results['round_robin']['u_deg'].append(urr[incident_server] / C[incident_server])
    results['round_robin']['migration'].append(np.linalg.norm(Xrr - state['round_robin']['X_prev'], 'fro'))
    results['round_robin']['latency_norm'].append(np.sum(Xrr * d_lat_norm) / N)
    results['round_robin']['load_std'].append(np.std(urr / C))
    results['round_robin']['stress_exposure'].append(np.sum((1 - CRSI) * urr))
    state['round_robin']['X_prev'] = Xrr

    Xg = greedy_least_loaded_assign(N, M, r, C)
    ug = Xg.T @ r
    results['greedy']['u_deg'].append(ug[incident_server] / C[incident_server])
    results['greedy']['migration'].append(np.linalg.norm(Xg - state['greedy']['X_prev'], 'fro'))
    results['greedy']['latency_norm'].append(np.sum(Xg * d_lat_norm) / N)
    results['greedy']['load_std'].append(np.std(ug / C))
    results['greedy']['stress_exposure'].append(np.sum((1 - CRSI) * ug))
    state['greedy']['X_prev'] = Xg

    Xth = crsi_threshold_heuristic(N, M, r, C, CRSI, theta=0.4)
    uth = Xth.T @ r
    results['crsi_threshold']['u_deg'].append(uth[incident_server] / C[incident_server])
    results['crsi_threshold']['migration'].append(np.linalg.norm(Xth - state['crsi_threshold']['X_prev'], 'fro'))
    results['crsi_threshold']['latency_norm'].append(np.sum(Xth * d_lat_norm) / N)
    results['crsi_threshold']['load_std'].append(np.std(uth / C))
    results['crsi_threshold']['stress_exposure'].append(np.sum((1 - CRSI) * uth))
    state['crsi_threshold']['X_prev'] = Xth

summary = {}
for m in methods:
    before = np.mean(results[m]['u_deg'][:incident_step])
    after = np.mean(results[m]['u_deg'][incident_step:])
    summary[m] = {
        'u_deg_before': before, 'u_deg_after': after,
        'pct_change': 100 * (after - before) / before,
        'mean_migration': np.mean(results[m]['migration']),
        'mean_latency_norm': np.mean(results[m]['latency_norm']),
        'mean_load_std': np.mean(results[m]['load_std']),
        'mean_stress_exposure': np.mean(results[m]['stress_exposure']),
    }
with open('../results/exp5_largescale.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
