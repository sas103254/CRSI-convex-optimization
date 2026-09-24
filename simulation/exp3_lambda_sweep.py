import os
import numpy as np, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import generate_crsi_components, crsi_from_components, solve_qp, crsi_weighted_round

M, N, T = 8, 60, 10
incident_server = 1
incident_step = 5
alpha, beta, gamma, delta = 0.30, 0.20, 0.20, 0.30

lambdas = [0.0, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0]
sweep = {}

for lam in lambdas:
    rng = np.random.default_rng(123)  # same seed per lambda for fair comparison
    C = rng.uniform(80, 120, M)
    e_energy = rng.uniform(0.008, 0.018, M)
    X_prev = np.ones((N, M)) / M
    prev_crsi_components = None
    migs, lats, loadstds = [], [], []
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
        Xc, st, _ = solve_qp(N, M, r, C, d_lat_norm, e_energy, CRSI, X_prev,
                              alpha, beta, gamma, delta, lam)
        Xd = crsi_weighted_round(Xc, CRSI)
        migs.append(np.linalg.norm(Xd - X_prev, 'fro'))
        lats.append(np.sum(Xd * d_lat_norm) / N)
        u = Xd.T @ r
        loadstds.append(np.std(u / C))
        X_prev = Xd
    sweep[str(lam)] = {
        'mean_migration': float(np.mean(migs)),
        'mean_latency_norm': float(np.mean(lats)),
        'mean_load_std': float(np.mean(loadstds)),
    }

with open('../results/exp3_lambda_sweep.json', 'w') as f:
    json.dump(sweep, f, indent=2)
print(json.dumps(sweep, indent=2))
