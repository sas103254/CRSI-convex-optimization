import os
import numpy as np, json, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import generate_crsi_components, crsi_from_components, solve_qp, crsi_weighted_round
import cvxpy as cp

alpha, beta, gamma, delta, lam = 0.30, 0.20, 0.20, 0.30, 0.15

sizes = [(8, 60), (16, 120), (32, 240), (64, 480), (128, 960)]
records = []

for M, N in sizes:
    rng = np.random.default_rng(2024)
    C = rng.uniform(80, 120, M)
    e_energy = rng.uniform(0.008, 0.018, M)
    r = rng.uniform(1, 10, N)
    d_latency = rng.uniform(1, 20, (N, M))
    d_lat_norm = d_latency / d_latency.mean()
    H, R, P, S = generate_crsi_components(M, rng)
    CRSI = crsi_from_components(H, R, P, S, (0.25, 0.25, 0.25, 0.25))
    X_prev = np.ones((N, M)) / M

    times = []
    for rep in range(3):
        Xc, st, _ = solve_qp(N, M, r, C, d_lat_norm, e_energy, CRSI, X_prev,
                              alpha, beta, gamma, delta, lam, solver=cp.CLARABEL)
        times.append(st)
    records.append({'M': M, 'N': N, 'vars': N * M,
                     'mean_solve_time_s': float(np.mean(times)),
                     'std_solve_time_s': float(np.std(times))})
    print(records[-1])

with open('../results/exp4_scalability.json', 'w') as f:
    json.dump(records, f, indent=2)
