import os
import numpy as np, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (generate_crsi_components, crsi_from_components, solve_qp,
                   crsi_weighted_round, objective_value)

rng = np.random.default_rng(7)
M, N, T = 8, 60, 10
incident_server = 1
incident_step = 5
C = rng.uniform(80, 120, M)
e_energy = rng.uniform(0.008, 0.018, M)
alpha, beta, gamma, delta, lam = 0.30, 0.20, 0.20, 0.30, 0.15

X_prev = np.ones((N, M)) / M
prev_crsi_components = None
gaps = []
infeas = []

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

    Xc, st, obj_c = solve_qp(N, M, r, C, d_lat_norm, e_energy, CRSI, X_prev,
                              alpha, beta, gamma, delta, lam)
    Xd = crsi_weighted_round(Xc, CRSI)

    obj_c_val = objective_value(Xc, r, C, d_lat_norm, e_energy, CRSI, X_prev,
                                 alpha, beta, gamma, delta, lam)
    obj_d_val = objective_value(Xd, r, C, d_lat_norm, e_energy, CRSI, X_prev,
                                 alpha, beta, gamma, delta, lam)

    # capacity feasibility check for the rounded (discrete) solution
    u_d = Xd.T @ r
    overflow = np.maximum(u_d - C, 0)
    infeas.append(float(np.sum(overflow)))

    gap_pct = 100.0 * (obj_d_val - obj_c_val) / abs(obj_c_val)
    gaps.append(gap_pct)
    X_prev = Xd

result = {
    'gap_pct_per_step': gaps,
    'mean_gap_pct': float(np.mean(gaps)),
    'max_gap_pct': float(np.max(gaps)),
    'capacity_overflow_per_step': infeas,
    'mean_capacity_overflow': float(np.mean(infeas)),
}
with open('../results/exp2_rounding_gap.json', 'w') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
