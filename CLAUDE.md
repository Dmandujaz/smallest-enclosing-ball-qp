# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mini-project for ITAM course MAT-24431 (Optimización Numérica I). Solves the **Smallest Enclosing Ball** problem by formulating it as a convex Quadratic Program (QP), following Schönherr (2002), Chapter 3.

## Commands

```bash
# Install dependencies
pip install numpy scipy cvxpy osqp pandas matplotlib

# Quick test (< 1 minute, 2 solvers, d=2..10, 5 trials)
python quick_demo.py

# Full experiment (d=2..30, 3 solvers, 20 trials, ~1740 QP solves, ~10-15 min)
python run_experiments.py

# Test solvers individually
python solvers.py

# Test data generation
python generate_data.py
```

## Architecture

Data flows: `generate_data.py` → `solvers.py` → `run_experiments.py`

### QP Formulation ([generate_data.py](generate_data.py))

`points_to_qp_format(points)` converts a (d×n) point matrix to QP form:
- **Q = 2CᵀC** (n×n, PSD, rank ≤ d — rank-deficient when d << n)
- **q_i = -‖pᵢ‖²**
- Constraint: **Σxᵢ = 1**, **x ≥ 0** (simplex)

`extract_solution(x_opt, points, obj_value)` recovers geometry:
- Center: **c* = Cx*** (weighted combination)
- Radius: **r* = √(-f(x*))**
- Active points: indices where xᵢ > 1e-6

### Solvers ([solvers.py](solvers.py))

All solvers share the same QP interface `(Q, q, A_eq, b_eq, bounds_lower)` and return a `SolverResult` dataclass. High-level entry point is `solve_smallest_enclosing_ball(points, solver, verbose)`.

| Solver name | Backend | Notes |
|---|---|---|
| `CVXPY-OSQP` | CVXPY + OSQP | Adds ε=1e-8 regularization to Q for `quad_form` |
| `CVXPY-SCS` | CVXPY + SCS | Same regularization |
| `scipy-SLSQP` | scipy | Fastest; uses dict-style constraint format |
| `scipy-trustconstr` | scipy | Provides Hessian; uses `LinearConstraint`/`Bounds` |
| `OSQP-direct` | OSQP (sparse) | No CVXPY overhead; stacks [A_eq; I] for bounds |

### Experiment Runner ([run_experiments.py](run_experiments.py))

- `run_single_experiment()` — one (n, d, solver, seed) combination; returns a flat dict
- `run_experiments()` — sweeps dimensions × trials × solvers; prints per-dimension summaries
- `compute_summary_statistics()` — groups by (dimension, solver), aggregates time/iters/n_active/error
- `save_results()` — writes `qp_results_raw.csv` and `qp_results_summary.csv` to output dir

Default output path: `/mnt/user-data/outputs/` (change `output_dir` arg for local runs).

## Key Theoretical Property

At optimum, **at most d+1 points are active** (xᵢ > 0), regardless of n. The experiment verifies this empirically. The `n_active` metric tracks this; violations are reported in `quick_demo.py`.

## Pending Work

- Visualization script (`visualize_results.py` referenced in README but not yet created)
- Statistical analysis of results
- 2-page final report
