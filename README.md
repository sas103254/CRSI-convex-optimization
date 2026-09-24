# Simulation & Paper-Generation Pipeline
### Adaptive Resource Allocation and Scheduling — CRSI Convex Optimization Framework

This is the complete, runnable pipeline used to generate every number, table, and
figure in the paper. It is organized into three stages:

```
pipeline/
├── simulation/            Stage 1 — core model + six experiments + figure generation
├── results/                Stage 1 output — JSON results, weight_log.npy, and figs/*.png
├── requirements.txt       minimum versions
└── requirements-lock.txt  exact versions used to verify reproduction
```

Stage 1 (`simulation/`) is the actual research pipeline — the convex-optimization
model, the adaptive weight controller, the baselines, and all six experiments.

## Requirements

- Python 3.10+
- `pip install -r requirements.txt` (numpy, cvxpy, matplotlib), or
  `pip install -r requirements-lock.txt` for the exact verified versions
- cvxpy will use the **CLARABEL** solver (installed automatically as a cvxpy
  dependency); no separate solver license or install step is needed.
- Stage 2 only: Python 3 (for `tex_to_parts.py`) and Node.js 18+ with the
  `docx` npm package (`npm install` inside `document_generation/`).
- Stage 3 only: a LaTeX distribution with the `IEEEtran` class (on
  Debian/Ubuntu: `apt install texlive-publishers`).

## Stage 1 — Reproducing the simulation results

All commands below are run from inside `simulation/`. Each experiment script
is self-contained (it seeds its own RNG) and writes its output as JSON into
`../results/`.

```bash
cd simulation
pip install -r ../requirements.txt

python3 exp1_main.py            # Primary scenario (M=8,N=60,T=10): all 6 methods,
                                 # incl. the adaptive weight controller.
                                 # -> ../results/exp1_results.json, exp1_summary.json,
                                 #    ../results/weight_log.npy

python3 exp2_rounding_gap.py    # Continuous-vs-rounded objective gap + capacity check.
                                 # -> ../results/exp2_rounding_gap.json

python3 exp3_lambda_sweep.py    # Regularization-weight (lambda) sensitivity sweep.
                                 # -> ../results/exp3_lambda_sweep.json

python3 exp4_scalability.py     # Solver runtime vs. problem size (480 to 122,880
                                 # decision variables). NOTE: the largest configuration
                                 # (M=128, N=960) takes ~20-25 seconds to solve.
                                 # -> ../results/exp4_scalability.json

python3 exp5_largescale.py      # Larger-scale validation (M=24, N=200, T=8).
                                 # -> ../results/exp5_largescale.json

python3 exp6_rounding_attribution.py
                                 # Separates the continuous QP solution from the
                                 # rounding step: degraded-server load in X* vs. the
                                 # rounded allocation, objective-term magnitudes, and a
                                 # plain-argmax rounding control (Sec. VI-I, VII-F).
                                 # -> ../results/exp6_rounding_attribution.json

python3 env_info.py             # Hardware/software report for Sec. VI-B. Run it on the
                                 # machine that produced the runtime numbers.
                                 # -> ../results/environment.json

python3 make_fig1.py            # Renders the architecture diagram (Fig. 1).
python3 make_figures.py         # Renders Figs. 2-9 from the JSON results above.
                                 # -> ../results/figs/*.png
```

Run them in this order (exp1 before make_figures, etc.) since the figure
scripts read the experiments' JSON output. `exp1_main.py` also prints its
summary dict to stdout when it finishes, so you can sanity-check numbers
immediately without opening the JSON.

### File-by-file description

| File | What it does |
|---|---|
| `core.py` | The model itself: CRSI component generation (`generate_crsi_components`, the AR(1) smoother), the convex QP (`solve_qp`, built with cvxpy, objective Eq. 6 + constraints Eqs. 7–9), CRSI-weighted rounding (`crsi_weighted_round`), the adaptive weight controller (`AdaptiveWeightController`, z-score → softmax → EMA), and the three baselines (`round_robin_assign`, `greedy_least_loaded_assign`, `crsi_threshold_heuristic`). |
| `exp1_main.py` | Main 10-step comparison across all 6 methods (fixed-weight proposed, adaptive-weight proposed, no-regularization ablation, Round-Robin, Greedy, CRSI-Threshold) with the synthetic stability incident injected at t=5. Produces Table III and Figs. 2–6. |
| `exp2_rounding_gap.py` | Re-solves the same scenario and compares the continuous QP objective J(X\*) against the discretized objective J(X_round) at every step, plus a capacity-feasibility check. Produces Fig. 7. |
| `exp3_lambda_sweep.py` | Re-runs the primary scenario at 9 values of λ (same random seed each time) to trace the migration/latency/load-balance trade-off curve. Produces Fig. 8. |
| `exp4_scalability.py` | Times the QP solve (3 repetitions each) at 5 problem sizes from (M=8,N=60) to (M=128,N=960). Produces Table IV and Fig. 9. |
| `exp5_largescale.py` | Repeats the stability-awareness comparison at 3x scale (M=24, N=200, T=8) with 4 methods. Produces Table V. |
| `exp6_rounding_attribution.py` | Replays the primary (seed 42) and larger-scale (seed 99) scenarios and records the degraded server's load in the continuous solution X* and in the rounded allocation, the magnitude of each objective term, the fraction of fractional tasks, and each task's share on its previous server; repeats the runs with plain argmax rounding. Produces the numbers in Sec. VI-I and VII-F. |
| `env_info.py` | Prints CPU model, cores/threads, clock, RAM, OS, and package versions, and saves them to `results/environment.json`. |
| `make_fig1.py` | Draws the architecture diagram (Fig. 1) with matplotlib (boxes/arrows), not derived from experiment JSON. |
| `make_figures.py` | Loads all the experiment JSON files and renders Figs. 2–9 (see the file-name table below). |

### Random seeds and settings

| Script | Seed | Notes |
|---|---|---|
| `exp1_main.py` | 42 | incident on server index 1 from t=5 (t=0..9); adaptive controller uses τ=0.5, **η=0.35** |
| `exp2_rounding_gap.py` | 7 | same configuration as exp1 |
| `exp3_lambda_sweep.py` | 123 | re-seeded at every λ |
| `exp4_scalability.py` | 2024 | re-seeded for every problem size; 3 timed solves per size |
| `exp5_largescale.py` | 99 | incident on server index 3 from t=4 (t=0..7) |
| `exp6_rounding_attribution.py` | 42, 99 | replays exp1 and exp5 exactly |


Each experiment script seeds its own `numpy.random.default_rng(...)` explicitly
(see the top of each file), so re-running any script reproduces the exact
numbers reported in the paper. `exp3_lambda_sweep.py` deliberately reuses the
*same* seed across every λ value so that the sweep isolates the effect of λ
alone.

## Notes on reproducibility

- All experiments use **synthetic** workloads (see Section VI-A of the paper
  for the exact distributions), not real cloud traces.
- `exp4_scalability.py`'s largest configuration is solver- and CPU-bound; on a
  slower machine it may take noticeably longer than ~20 seconds.
- If `cvxpy` fails to find a solver, `pip install cvxpy` should pull in
  CLARABEL automatically; `core.solve_qp` falls back to SCS if CLARABEL fails
  to return a solution. The fallback is not triggered in any experiment.
- Re-running exp1, exp2, exp3, and exp5 with the versions in
  `requirements-lock.txt` reproduces every non-timing value in `results/` to
  within 1e-6. Solve times depend on the machine; the values in `results/`
  come from the machine described in Sec. VI-B of the paper.
- In `core.solve_qp`, the latency and energy terms are normalized (latency by
  its per-step mean, energy via u_j/C_j), while the load-balancing term
  `sum_squares(u)` and the stability term use raw utilization. The load term is
  therefore about two orders of magnitude larger than the others, and the
  degraded-server load reduction comes mainly from `crsi_weighted_round`
  (see `exp6_rounding_attribution.py` and Sec. VI-I / VII-F of the paper).

## Figure file names

| Paper figure | File in `results/figs/` |
|---|---|
| Fig. 1 architecture | `fig1_architecture.png` |
| Fig. 2 load shed | `fig2_load_shed.png` |
| Fig. 3 adaptive weights | `fig6_adaptive_weights.png` |
| Fig. 4 migration | `fig3_migration.png` |
| Fig. 5 stress exposure | `fig4_stress_exposure.png` |
| Fig. 6 load balance | `fig5_load_balance.png` |
| Fig. 7 rounding gap | `fig9_rounding_gap.png` |
| Fig. 8 λ sensitivity | `fig7_lambda_sensitivity.png` |
| Fig. 9 solver runtime | `fig8_scalability.png` |
