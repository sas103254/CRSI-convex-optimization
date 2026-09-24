"""
Experiment 6: where does the degraded server's load reduction come from?

Replays the primary scenario of exp1_main.py (same seed, same draws, same
incident) for the three QP-based variants, and the larger-scale scenario of
exp5_largescale.py for the proposed method, and records, at every step:

  * the degraded server's normalized load u_j/C_j in the CONTINUOUS
    solution X* (before rounding) and in the ROUNDED allocation;
  * the magnitude of each objective term at X* (load, energy, latency,
    stability, regularization);
  * the mean continuous share x*_ij that each task keeps on the server it
    was assigned to at t-1.

It also re-runs the fixed-weight and adaptive-weight variants with plain
argmax rounding (no CRSI weighting) to isolate the effect of the
(0.5 + 0.5*CRSI_j) factor in crsi_weighted_round.

The baselines in exp1/exp5 do not draw from the RNG, so omitting them here
leaves the random streams identical; the rounded loads reproduce exp1's
-50.0% (fixed) and -3.0% (adaptive) and exp5's -30.1% load-shed values.

Output: ../results/exp6_rounding_attribution.json  (used in Sec. VI-I)
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (generate_crsi_components, crsi_from_components, solve_qp,
                  crsi_weighted_round, AdaptiveWeightController)

# Scenario configurations mirror exp1_main.py and exp5_largescale.py exactly.
CONFIGS = {
    'primary':     dict(seed=42, M=8,  N=60,  T=10, incident_server=1, incident_step=5,
                        variants=['proposed_fixed', 'proposed_adaptive', 'no_reg']),
    'large_scale': dict(seed=99, M=24, N=200, T=8,  incident_server=3, incident_step=4,
                        variants=['proposed_fixed']),
}
FIXED_W = (0.30, 0.20, 0.20, 0.30)
LAM = 0.15
ETA = 0.35  # same as exp1_main.py


def plain_round(Xc):
    Xd = np.zeros_like(Xc)
    Xd[np.arange(Xc.shape[0]), np.argmax(Xc, axis=1)] = 1.0
    return Xd


def run(rounding, cfg):
    M, N, T = cfg['M'], cfg['N'], cfg['T']
    INCIDENT_SERVER, INCIDENT_STEP = cfg['incident_server'], cfg['incident_step']
    rng = np.random.default_rng(cfg['seed'])
    C = rng.uniform(80, 120, M)
    e_energy = rng.uniform(0.008, 0.018, M)

    variants = cfg['variants']
    X_prev = {v: np.ones((N, M)) / M for v in variants}
    ctrl = AdaptiveWeightController(w0=FIXED_W, eta=ETA)
    prev_comp = None

    rec = {v: {'u_deg_continuous': [], 'u_deg_rounded': [], 'share_on_prev_server': [],
               'term_load': [], 'term_energy': [], 'term_latency': [],
               'term_stability': [], 'term_reg': [], 'frac_tasks_fractional': [],
               'crsi_degraded': []} for v in variants}

    for t in range(T):
        r = rng.uniform(1, 10, N)
        d = rng.uniform(1, 20, (N, M)) + rng.normal(0, 1.0, (N, M))
        d = np.clip(d, 0.5, None)
        H, R, P, S = generate_crsi_components(M, rng, prev_state=prev_comp,
                                              incident_server=INCIDENT_SERVER,
                                              incident_active=(t >= INCIDENT_STEP))
        prev_comp = {'H': H, 'R': R, 'P': P, 'S': S}
        d_norm = d / d.mean()
        CRSI = crsi_from_components(H, R, P, S, (0.25, 0.25, 0.25, 0.25))

        for v in variants:
            if v == 'proposed_adaptive':
                w, lam = tuple(ctrl.w), LAM
            elif v == 'no_reg':
                w, lam = FIXED_W, 0.0
            else:
                w, lam = FIXED_W, LAM
            a, b, g, dl = w
            Xc, _, _ = solve_qp(N, M, r, C, d_norm, e_energy, CRSI, X_prev[v], a, b, g, dl, lam)
            Xd = crsi_weighted_round(Xc, CRSI) if rounding == 'crsi' else plain_round(Xc)

            uc = Xc.T @ r
            ud = Xd.T @ r
            prev_j = np.argmax(X_prev[v], axis=1)
            rec[v]['u_deg_continuous'].append(float(uc[INCIDENT_SERVER] / C[INCIDENT_SERVER]))
            rec[v]['u_deg_rounded'].append(float(ud[INCIDENT_SERVER] / C[INCIDENT_SERVER]))
            # t=0 starts from the uniform allocation, where "previous server" is undefined
            rec[v]['share_on_prev_server'].append(
                float(Xc[np.arange(N), prev_j].mean()) if t > 0 else None)
            # a task is "fractional" if no single server holds >= 0.99 of it in X*
            rec[v]['frac_tasks_fractional'].append(float((Xc.max(axis=1) < 0.99).mean()))
            rec[v]['crsi_degraded'].append(float(CRSI[INCIDENT_SERVER]))
            rec[v]['term_load'].append(float(a * np.sum(uc ** 2)))
            rec[v]['term_energy'].append(float(b * np.sum(e_energy * (uc / C) ** 2)))
            rec[v]['term_latency'].append(float(g * np.sum(Xc * d_norm)))
            rec[v]['term_stability'].append(float(dl * np.sum((1 - CRSI) * uc)))
            rec[v]['term_reg'].append(float(lam * np.sum((Xc - X_prev[v]) ** 2)))

            if v == 'proposed_adaptive':
                ctrl.update(np.std(ud / C) / (np.mean(ud / C) + 1e-9),
                            np.mean(e_energy * (ud / C) ** 2),
                            np.mean(Xd * d_norm),
                            np.max(1 - CRSI))
            X_prev[v] = Xd

    summary = {}
    for v in variants:
        ud = np.array(rec[v]['u_deg_rounded'])
        before, after = ud[:INCIDENT_STEP].mean(), ud[INCIDENT_STEP:].mean()
        sh = rec[v]['share_on_prev_server']
        summary[v] = {
            'load_shed_pct_rounded': float(100 * (after - before) / before),
            'u_deg_continuous_min': float(np.min(rec[v]['u_deg_continuous'])),
            'u_deg_continuous_max': float(np.max(rec[v]['u_deg_continuous'])),
            'share_on_prev_pre_incident': float(np.mean(sh[1:INCIDENT_STEP])),
            'share_on_prev_post_incident': float(np.mean(sh[INCIDENT_STEP:])),
            'term_load_range': [float(min(rec[v]['term_load'])), float(max(rec[v]['term_load']))],
            'term_stability_max': float(max(rec[v]['term_stability'])),
            'mean_frac_tasks_fractional': float(np.mean(rec[v]['frac_tasks_fractional'])),
            'crsi_degraded_pre_mean': float(np.mean(rec[v]['crsi_degraded'][:INCIDENT_STEP])),
            'crsi_degraded_post_mean': float(np.mean(rec[v]['crsi_degraded'][INCIDENT_STEP:])),
        }
    return {'per_step': rec, 'summary': summary}


if __name__ == '__main__':
    out = {name: {'crsi_weighted_rounding': run('crsi', cfg),
                  'plain_argmax_rounding': run('plain', cfg)}
           for name, cfg in CONFIGS.items()}
    with open('../results/exp6_rounding_attribution.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({n: {k: v['summary'] for k, v in o.items()} for n, o in out.items()}, indent=2))
