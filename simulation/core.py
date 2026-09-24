"""
Core simulation module for the Adaptive Resource Allocation and
CRSI-based Convex Scheduling Framework (revised version).

Implements:
  - Explicit CRSI component generation (H, R, P, S) with normalization
  - The strictly convex QP (objective (5) in the paper) solved via cvxpy
  - CRSI-weighted max-argument discretization/rounding
  - An adaptive weight controller for (alpha, beta, gamma, delta)
  - Two baselines: Round-Robin and Greedy-Least-Loaded
  - A new, stronger baseline: CRSI-Threshold Heuristic (rule-based,
    CRSI-aware, but non-optimization)
"""
import numpy as np
import cvxpy as cp
import time


# ----------------------------------------------------------------------
# CRSI component generation
# ----------------------------------------------------------------------
def generate_crsi_components(M, rng, prev_state=None, incident_server=None,
                              incident_active=False):
    """
    Generate the four CRSI sub-scores for M servers at one time step.
    All returned in [0, 1], higher = healthier, consistent with the
    (1 - CRSI_j) penalty structure used in the objective.

    Each component is modeled as a bounded, mean-reverting AR(1) process
    driven by a proxy physical signal, so that CRSI evolves smoothly
    rather than being drawn i.i.d. at every step (which would make the
    "stability" framing meaningless).

        H_j (thermal):     1 - clip(temp_j / temp_max)
        R_j (utilization):  1 - clip(mem_util_j)
        P_j (latency/pkt):  1 - clip(w_p1 * pkt_drop_j + w_p2 * norm_latency_j)
        S_j (security):     1 - clip(threat_score_j)

    prev_state: dict with keys 'H','R','P','S' (arrays) from t-1, or None
                for the first step (initializes near 0.8, i.e. healthy).
    """
    if prev_state is None:
        H = rng.uniform(0.7, 0.9, M)
        R = rng.uniform(0.7, 0.9, M)
        P = rng.uniform(0.7, 0.9, M)
        S = rng.uniform(0.7, 0.9, M)
    else:
        # Mean-reverting update: x_t = phi*x_{t-1} + (1-phi)*mu + noise
        phi = 0.85
        mu = 0.8
        noise_scale = 0.03
        H = phi * prev_state['H'] + (1 - phi) * mu + rng.normal(0, noise_scale, M)
        R = phi * prev_state['R'] + (1 - phi) * mu + rng.normal(0, noise_scale, M)
        P = phi * prev_state['P'] + (1 - phi) * mu + rng.normal(0, noise_scale, M)
        S = phi * prev_state['S'] + (1 - phi) * mu + rng.normal(0, noise_scale, M)
        H, R, P, S = (np.clip(x, 0.02, 0.98) for x in (H, R, P, S))

    if incident_active and incident_server is not None:
        # Synthetic stability incident: thermal + link-quality + security
        # co-degrade sharply on the affected server (models e.g. a fan
        # failure combined with a noisy-neighbor / intrusion event).
        H[incident_server] = 0.15
        P[incident_server] = 0.25
        S[incident_server] = 0.30
        R[incident_server] = 0.35

    return np.clip(H, 0, 1), np.clip(R, 0, 1), np.clip(P, 0, 1), np.clip(S, 0, 1)


def crsi_from_components(H, R, P, S, w):
    w1, w2, w3, w4 = w
    return w1 * H + w2 * R + w3 * P + w4 * S


# ----------------------------------------------------------------------
# Adaptive weight controller (addresses Fig. 1's "Adaptive Weight
# Controller" block, previously unimplemented)
# ----------------------------------------------------------------------
class AdaptiveWeightController:
    """
    Online softmax controller for (alpha, beta, gamma, delta) that
    directly implements the "Adaptive Weight Controller" block of
    Fig. 1 (previously present only in the architecture diagram, not
    in the evaluated model).

    Each objective term is scored by a signal that reflects how much
    the *current* system state needs that term emphasized:

      score_alpha (load)     = std(u_j/C_j) / mean(u_j/C_j)   [imbalance]
      score_beta  (energy)   = mean_j normalized energy cost
      score_gamma (latency)  = mean normalized latency of the allocation
      score_delta (stability)= max_j (1 - CRSI_j)             [worst-case
                                stress, so a single degraded server is
                                visible even when M is large]

    Signals are z-scored against a running mean/std (not against each
    other within one step) so one term cannot mechanically dominate by
    construction, and a temperature-scaled softmax maps the score
    vector directly onto the probability simplex:

        w_target = softmax(z_scores / tau)

    The weights actually used are an exponential moving average of the
    previous weights and this target,
        w_new = (1 - eta) * w_prev + eta * w_target,
    which damps oscillation (paralleling the role that lambda plays for
    migration) while still tracking sustained regime changes such as a
    stability incident.
    """

    def __init__(self, w0=(0.30, 0.20, 0.20, 0.30), eta=0.30, tau=0.5, w_min=0.05):
        self.w = np.array(w0, dtype=float)
        self.eta = eta
        self.tau = tau
        self.w_min = w_min
        self.history = [self.w.copy()]
        # running stats for z-scoring, one per signal, seeded with mild priors
        self._mean = np.array([0.3, 0.01, 0.5, 0.3])
        self._var = np.array([0.05, 0.001, 0.05, 0.05])
        self._n = 1

    def _zscore_update(self, x):
        # Welford-style running mean/var, then z-score the new sample
        self._n += 1
        delta = x - self._mean
        self._mean = self._mean + delta / self._n
        self._var = self._var + delta * (x - self._mean)
        std = np.sqrt(np.maximum(self._var / self._n, 1e-6))
        return (x - self._mean) / std

    def update(self, load_imbalance, energy_signal, latency_signal, stability_signal):
        raw = np.array([load_imbalance, energy_signal, latency_signal, stability_signal])
        z = self._zscore_update(raw)
        z = np.clip(z, -3, 3)
        exp_z = np.exp(z / self.tau)
        w_target = exp_z / exp_z.sum()
        w_target = np.clip(w_target, self.w_min, None)
        w_target = w_target / w_target.sum()
        self.w = (1 - self.eta) * self.w + self.eta * w_target
        self.w = self.w / self.w.sum()
        self.history.append(self.w.copy())
        return self.w


# ----------------------------------------------------------------------
# Convex QP solver (objective (5), constraints (6)-(8))
# ----------------------------------------------------------------------
def solve_qp(N, M, r, C, d_latency, e_energy, CRSI, X_prev,
             alpha, beta, gamma, delta, lam, solver=cp.CLARABEL):
    """
    Solve the strictly convex relaxation:
        min_X sum_j [ alpha*u_j^2 + beta*E_j(u_j) + gamma*L_j(X)
                      + delta*(1-CRSI_j)*u_j ] + lam*||X - X_prev||_F^2
        s.t.  sum_j x_ij = 1  for all i
              u_j = sum_i x_ij r_i <= C_j
              0 <= x_ij <= 1

    L_j(X) is modeled as sum_i x_ij * d_ij (expected communication
    latency contributed to server j), consistent with Table II.
    Returns (X_opt, solve_time_seconds, objective_value).
    """
    X = cp.Variable((N, M))
    u = X.T @ r  # server utilizations, shape (M,)

    energy_term = cp.sum([e_energy[j] * cp.square(u[j] / C[j]) for j in range(M)])
    latency_term = cp.sum(cp.multiply(X, d_latency))  # sum_i sum_j x_ij d_ij
    load_term = cp.sum_squares(u)
    stability_term = cp.sum(cp.multiply((1 - CRSI), u))
    reg_term = cp.sum_squares(X - X_prev)

    objective = (alpha * load_term + beta * energy_term + gamma * latency_term
                 + delta * stability_term + lam * reg_term)

    constraints = [
        cp.sum(X, axis=1) == 1,
        u <= C,
        X >= 0,
        X <= 1,
    ]

    prob = cp.Problem(cp.Minimize(objective), constraints)
    t0 = time.time()
    prob.solve(solver=solver)
    t1 = time.time()
    Xval = X.value
    if Xval is None:
        # fall back solver
        prob.solve(solver=cp.SCS)
        Xval = X.value
    return Xval, (t1 - t0), prob.value


def crsi_weighted_round(Xc, CRSI):
    """
    Phase-4 discretization: CRSI-weighted max-argument rounding.
    For each task i, assign it fully to server
        j* = argmax_j  x_ij * (0.5 + 0.5*CRSI_j)
    i.e. ties among near-equal continuous shares are broken in favor of
    healthier servers, rather than pure argmax over x_ij alone.
    """
    N, M = Xc.shape
    score = Xc * (0.5 + 0.5 * CRSI[np.newaxis, :])
    choice = np.argmax(score, axis=1)
    Xd = np.zeros_like(Xc)
    Xd[np.arange(N), choice] = 1.0
    return Xd


def objective_value(X, r, C, d_latency, e_energy, CRSI, X_prev,
                     alpha, beta, gamma, delta, lam):
    u = X.T @ r
    energy_term = np.sum(e_energy * (u / C) ** 2)
    latency_term = np.sum(X * d_latency)
    load_term = np.sum(u ** 2)
    stability_term = np.sum((1 - CRSI) * u)
    reg_term = np.sum((X - X_prev) ** 2)
    return (alpha * load_term + beta * energy_term + gamma * latency_term
            + delta * stability_term + lam * reg_term)


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------
def round_robin_assign(N, M, r, C, u_prev_load, rr_pointer):
    X = np.zeros((N, M))
    load = np.zeros(M)
    ptr = rr_pointer
    for i in range(N):
        # cycle, but skip a server if it would blow capacity (basic feasibility)
        tries = 0
        while load[ptr] + r[i] > C[ptr] and tries < M:
            ptr = (ptr + 1) % M
            tries += 1
        X[i, ptr] = 1.0
        load[ptr] += r[i]
        ptr = (ptr + 1) % M
    return X, ptr


def greedy_least_loaded_assign(N, M, r, C):
    X = np.zeros((N, M))
    load = np.zeros(M)
    order = np.argsort(-r)  # place larger tasks first (common greedy practice)
    for i in order:
        # choose least relatively loaded feasible server
        rel_load = load / C
        feasible = np.where(load + r[i] <= C)[0]
        if len(feasible) == 0:
            j = np.argmin(load)  # overflow fallback
        else:
            j = feasible[np.argmin(rel_load[feasible])]
        X[i, j] = 1.0
        load[j] += r[i]
    return X


def crsi_threshold_heuristic(N, M, r, C, CRSI, theta=0.4):
    """
    NEW, stronger baseline: CRSI-Threshold Heuristic.
    Rule-based (no optimization): servers with CRSI_j < theta are
    excluded from consideration unless every server is degraded; among
    the remaining eligible servers, tasks are placed greedily onto the
    least relatively-loaded one. This isolates how much of the proposed
    method's stability-awareness can be captured by a simple threshold
    rule without a convex program or migration control.
    """
    eligible = np.where(CRSI >= theta)[0]
    if len(eligible) == 0:
        eligible = np.arange(M)
    X = np.zeros((N, M))
    load = np.zeros(M)
    order = np.argsort(-r)
    for i in order:
        rel_load = load / C
        cand = eligible
        feasible = cand[load[cand] + r[i] <= C[cand]]
        if len(feasible) == 0:
            feasible = np.where(load + r[i] <= C)[0]
            if len(feasible) == 0:
                feasible = np.array([np.argmin(load)])
        j = feasible[np.argmin(rel_load[feasible])]
        X[i, j] = 1.0
        load[j] += r[i]
    return X
