import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (generate_crsi_components, crsi_from_components, solve_qp,
                   crsi_weighted_round, objective_value, round_robin_assign,
                   greedy_least_loaded_assign, crsi_threshold_heuristic,
                   AdaptiveWeightController)

rng = np.random.default_rng(42)

M, N, T = 8, 60, 10
incident_server = 1  # "Server 2" (0-indexed as 1)
incident_step = 5

C = rng.uniform(80, 120, M)
e_energy = rng.uniform(0.008, 0.018, M)

methods = ['proposed_fixed', 'proposed_adaptive', 'no_reg', 'round_robin',
           'greedy', 'crsi_threshold']

state = {m: {} for m in methods}
for m in methods:
    state[m]['X_prev'] = np.ones((N, M)) / M
    state[m]['rr_ptr'] = 0
    state[m]['crsi_prev_components'] = None

results = {m: {'u_deg': [], 'migration': [], 'latency_norm': [], 'load_std': [],
               'stress_exposure': [], 'solve_time': [], 'load_deg_series': [],
               'energy_norm': []}
           for m in methods}

adaptive_ctrl = AdaptiveWeightController(w0=(0.30, 0.20, 0.20, 0.30), eta=0.35)
weight_log = []

FIXED_W = (0.30, 0.20, 0.20, 0.30)  # alpha, beta, gamma, delta
LAM = 0.15

prev_crsi_components = None

for t in range(T):
    r = rng.uniform(1, 10, N)
    d_latency = rng.uniform(1, 20, (N, M)) + rng.normal(0, 1.0, (N, M))
    d_latency = np.clip(d_latency, 0.5, None)

    incident_active = (t >= incident_step)
    H, R, P, S = generate_crsi_components(M, rng, prev_state=prev_crsi_components,
                                           incident_server=incident_server,
                                           incident_active=incident_active)
    prev_crsi_components = {'H': H, 'R': R, 'P': P, 'S': S}

    # normalize latency/energy by capacity & mean latency for dimensional comparability
    mean_lat = d_latency.mean()
    d_lat_norm = d_latency / mean_lat

    w_crsi = (0.25, 0.25, 0.25, 0.25)
    CRSI = crsi_from_components(H, R, P, S, w_crsi)

    # ---- Proposed (fixed weights + regularization) ----
    a, b, g, dl = FIXED_W
    Xc, st, _ = solve_qp(N, M, r, C, d_lat_norm, e_energy, CRSI,
                          state['proposed_fixed']['X_prev'], a, b, g, dl, LAM)
    Xd = crsi_weighted_round(Xc, CRSI)
    u = Xd.T @ r
    results['proposed_fixed']['u_deg'].append(u[incident_server] / C[incident_server])
    results['proposed_fixed']['migration'].append(
        np.linalg.norm(Xd - state['proposed_fixed']['X_prev'], 'fro'))
    results['proposed_fixed']['latency_norm'].append(np.sum(Xd * d_lat_norm) / N)
    results['proposed_fixed']['load_std'].append(np.std(u / C))
    results['proposed_fixed']['stress_exposure'].append(np.sum((1 - CRSI) * u))
    results['proposed_fixed']['energy_norm'].append(np.mean(e_energy * (u / C) ** 2))
    results['proposed_fixed']['solve_time'].append(st)
    state['proposed_fixed']['X_prev'] = Xd

    # ---- Proposed (ADAPTIVE weights + regularization) [NEW] ----
    a2, b2, g2, dl2 = adaptive_ctrl.w
    Xc2, st2, _ = solve_qp(N, M, r, C, d_lat_norm, e_energy, CRSI,
                            state['proposed_adaptive']['X_prev'], a2, b2, g2, dl2, LAM)
    Xd2 = crsi_weighted_round(Xc2, CRSI)
    u2 = Xd2.T @ r
    results['proposed_adaptive']['u_deg'].append(u2[incident_server] / C[incident_server])
    results['proposed_adaptive']['migration'].append(
        np.linalg.norm(Xd2 - state['proposed_adaptive']['X_prev'], 'fro'))
    results['proposed_adaptive']['latency_norm'].append(np.sum(Xd2 * d_lat_norm) / N)
    results['proposed_adaptive']['load_std'].append(np.std(u2 / C))
    results['proposed_adaptive']['stress_exposure'].append(np.sum((1 - CRSI) * u2))
    results['proposed_adaptive']['energy_norm'].append(np.mean(e_energy * (u2 / C) ** 2))
    results['proposed_adaptive']['solve_time'].append(st2)
    state['proposed_adaptive']['X_prev'] = Xd2

    # feed signals to the controller AFTER using this step's weights
    load_imbalance = np.std(u2 / C) / (np.mean(u2 / C) + 1e-9)
    energy_signal = np.mean(e_energy * (u2 / C) ** 2)
    latency_signal = np.mean(Xd2 * d_lat_norm)
    stability_signal = np.max(1 - CRSI)
    new_w = adaptive_ctrl.update(load_imbalance, energy_signal, latency_signal, stability_signal)
    weight_log.append(new_w.copy())

    # ---- Proposed (no regularization, lambda=0) ----
    Xc3, st3, _ = solve_qp(N, M, r, C, d_lat_norm, e_energy, CRSI,
                            state['no_reg']['X_prev'], a, b, g, dl, 0.0)
    Xd3 = crsi_weighted_round(Xc3, CRSI)
    u3 = Xd3.T @ r
    results['no_reg']['u_deg'].append(u3[incident_server] / C[incident_server])
    results['no_reg']['migration'].append(np.linalg.norm(Xd3 - state['no_reg']['X_prev'], 'fro'))
    results['no_reg']['latency_norm'].append(np.sum(Xd3 * d_lat_norm) / N)
    results['no_reg']['load_std'].append(np.std(u3 / C))
    results['no_reg']['stress_exposure'].append(np.sum((1 - CRSI) * u3))
    results['no_reg']['energy_norm'].append(np.mean(e_energy * (u3 / C) ** 2))
    results['no_reg']['solve_time'].append(st3)
    state['no_reg']['X_prev'] = Xd3

    # ---- Round Robin ----
    Xrr, new_ptr = round_robin_assign(N, M, r, C, None, state['round_robin']['rr_ptr'])
    state['round_robin']['rr_ptr'] = new_ptr
    urr = Xrr.T @ r
    results['round_robin']['u_deg'].append(urr[incident_server] / C[incident_server])
    results['round_robin']['migration'].append(np.linalg.norm(Xrr - state['round_robin']['X_prev'], 'fro'))
    results['round_robin']['latency_norm'].append(np.sum(Xrr * d_lat_norm) / N)
    results['round_robin']['load_std'].append(np.std(urr / C))
    results['round_robin']['stress_exposure'].append(np.sum((1 - CRSI) * urr))
    results['round_robin']['energy_norm'].append(np.mean(e_energy * (urr / C) ** 2))
    results['round_robin']['solve_time'].append(0.0)
    state['round_robin']['X_prev'] = Xrr

    # ---- Greedy Least Loaded ----
    Xg = greedy_least_loaded_assign(N, M, r, C)
    ug = Xg.T @ r
    results['greedy']['u_deg'].append(ug[incident_server] / C[incident_server])
    results['greedy']['migration'].append(np.linalg.norm(Xg - state['greedy']['X_prev'], 'fro'))
    results['greedy']['latency_norm'].append(np.sum(Xg * d_lat_norm) / N)
    results['greedy']['load_std'].append(np.std(ug / C))
    results['greedy']['stress_exposure'].append(np.sum((1 - CRSI) * ug))
    results['greedy']['energy_norm'].append(np.mean(e_energy * (ug / C) ** 2))
    results['greedy']['solve_time'].append(0.0)
    state['greedy']['X_prev'] = Xg

    # ---- CRSI-Threshold Heuristic [NEW baseline] ----
    Xth = crsi_threshold_heuristic(N, M, r, C, CRSI, theta=0.4)
    uth = Xth.T @ r
    results['crsi_threshold']['u_deg'].append(uth[incident_server] / C[incident_server])
    results['crsi_threshold']['migration'].append(np.linalg.norm(Xth - state['crsi_threshold']['X_prev'], 'fro'))
    results['crsi_threshold']['latency_norm'].append(np.sum(Xth * d_lat_norm) / N)
    results['crsi_threshold']['load_std'].append(np.std(uth / C))
    results['crsi_threshold']['stress_exposure'].append(np.sum((1 - CRSI) * uth))
    results['crsi_threshold']['energy_norm'].append(np.mean(e_energy * (uth / C) ** 2))
    results['crsi_threshold']['solve_time'].append(0.0)
    state['crsi_threshold']['X_prev'] = Xth

# ---- save summary ----
summary = {}
for m in methods:
    before = np.mean(results[m]['u_deg'][:incident_step])
    after = np.mean(results[m]['u_deg'][incident_step:])
    summary[m] = {
        'u_deg_before': before,
        'u_deg_after': after,
        'pct_change': 100 * (after - before) / before,
        'mean_migration': np.mean(results[m]['migration']),
        'mean_latency_norm': np.mean(results[m]['latency_norm']),
        'mean_load_std': np.mean(results[m]['load_std']),
        'mean_stress_exposure': np.mean(results[m]['stress_exposure']),
        'mean_solve_time_ms': np.mean(results[m]['solve_time']) * 1000,
        'mean_energy_norm': np.mean(results[m]['energy_norm']),
    }

with open('../results/exp1_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

weight_log = np.array(weight_log)
np.save('../results/weight_log.npy', weight_log)

for m in methods:
    for k in results[m]:
        results[m][k] = list(np.array(results[m][k]).astype(float))
with open('../results/exp1_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(summary, indent=2))
