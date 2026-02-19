# Smallest Enclosing Ball via Quadratic Programming

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fast, efficient solver for the **Smallest Enclosing Ball** (Minimum Enclosing Sphere) problem using convex quadratic programming. Compares multiple QP solvers (CVXPY-OSQP, scipy-SLSQP, OSQP-direct) across dimensions 2-30 with rigorous benchmarking.

## Quick Start

```bash
# Install dependencies
pip install numpy scipy cvxpy osqp pandas

# Run quick demo (< 1 min, d=2-10)
python quick_demo.py

# Run full experiments (10-15 min, d=2-30, 1740 problems)
python run_experiments.py
```

---

## Problem Statement

### Geometric Problem

Given **n points** {p₁, ..., pₙ} in ℝᵈ, find:
- Center c* ∈ ℝᵈ
- Radius r* ∈ ℝ

Such that:
1. All points are enclosed: ‖pᵢ - c*‖ ≤ r* for all i
2. Radius r* is minimized

### Quadratic Programming Formulation

Following Schönherr (2002, Theorem 3.1), the problem is reformulated as:

```
minimize    x^T C^T C x - Σᵢ ‖pᵢ‖² xᵢ
subject to  Σᵢ xᵢ = 1
            x ≥ 0
```

Where:
- C = [p₁ | p₂ | ... | pₙ] ∈ ℝᵈˣⁿ is the point matrix
- x ∈ ℝⁿ are convex combination weights

**Solution Recovery**:
- Center: c* = Σᵢ pᵢ xᵢ* (weighted combination of points)
- Radius: r* = √(-f(x*)) (from optimal objective value)

**Key Theoretical Property**: At optimality, **at most d+1 points are active** (xᵢ* > 0), regardless of n. This is empirically verified in the experiments.

---

## Project Structure

```
./qp-solver/
├── generate_data.py         # Random data generation and QP format conversion
├── solvers.py               # Multiple QP solver implementations
├── run_experiments.py       # Main experimental harness
├── quick_demo.py            # Quick demo for testing
├── README.md                # This file
└── Final_opti.pdf           # Detailed report (Spanish)
```

---

## Experiment Configuration

### Fixed Parameters
- **n = 50 points** (constant across all dimensions)
- **Distribution**: Uniform on [-1, 1]ᵈ
- **Trials per dimension**: 20 random instances

### Dimensions Tested
- **d = 2, 3, 4, ..., 30** (29 dimensions total)

### Solvers Compared

| Solver | Backend | Speed | Notes |
|--------|---------|-------|-------|
| **CVXPY-OSQP** | CVXPY + OSQP | Medium | Robust, recommended for full experiments |
| **scipy-SLSQP** | SciPy | Fast | Fastest overall, excellent convergence |
| **OSQP-direct** | OSQP (sparse) | Fast | Low overhead, good for high dimensions |
| CVXPY-SCS | CVXPY + SCS | Slow | For comparison |
| scipy-trustconstr | SciPy | Medium | Trust-region method |

---

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/Dmandujaz/smallest-enclosing-ball-qp.git
cd qp-solver

# Install dependencies
pip install numpy scipy cvxpy osqp pandas
```

### Quick Test (< 1 minute)

```bash
python quick_demo.py
```

Tests dimensions d=2-10 with 5 trials to verify the setup works.

### Full Experiments (10-15 minutes)

```bash
python run_experiments.py
```

Runs:
- 29 dimensions (d=2 through d=30)
- 20 trials per dimension
- 3 solvers (CVXPY-OSQP, scipy-SLSQP, OSQP-direct)
- **Total: 1,740 QP problems**

Results are saved to:
- `qp_results_raw.csv` - All 1,740 individual results
- `qp_results_summary.csv` - Aggregated statistics by dimension/solver

### Programmatic Usage

```python
from run_experiments import run_experiments

results = run_experiments(
    dimensions=[2, 5, 10],
    n_points=50,
    n_trials=5,
    solvers=['CVXPY-OSQP', 'scipy-SLSQP']
)
# Returns pandas DataFrame with all results
```

---

## Metrics Collected

For each problem instance:

| Metric | Type | Description |
|--------|------|-------------|
| **solve_time** | float | Execution time in seconds |
| **n_iters** | int | Number of iterations |
| **obj_value** | float | Objective value from QP |
| **radius** | float | Radius of enclosing ball |
| **n_active_pts** | int | Number of points with xᵢ > 1e-6 |
| **verify_error** | float | \|max_distance - radius\| |
| **solver_status** | str | 'success' or error message |

---

## Research Questions

This project investigates:

1. **Computational Scaling**: How do solve time and iterations scale with dimension d?
2. **Solver Performance**: Which solver is fastest? Most robust? Most accurate?
3. **Theoretical Verification**: Does the d+1 active point property hold empirically?
4. **Convergence Behavior**: Are there dimension thresholds where convergence fails?
5. **Rank Deficiency**: How does Q's rank-deficiency (rank ≤ d, n=50 fixed) affect solvers?

**Expected Results**:
- Time should be roughly **constant in d** (since n=50 is fixed)
- Iterations may decrease with d (better conditioning at higher rank)
- scipy-SLSQP expected to be fastest (no framework overhead)
- Active points should not exceed d+1 at optimality

---

## References

- **Schönherr, J.** (2002). *Smooth Geometry for Convex Hull Computation*.
  PhD thesis, ETH Zürich. Chapter 3: Geometric Optimization Problems.
  [[PDF]](schoenherr02.pdf)

- **Nocedal, J., Wright, S. J.** (2006). *Numerical Optimization* (2nd ed.).
  Springer. Chapter 16: Quadratic Programming.

- **CVXPY** - Convex optimization in Python: https://www.cvxpy.org/
- **OSQP** - Operator Splitting Quadratic Program solver: https://osqp.org/

---

## Dependencies

- **Python** 3.10+
- **NumPy** 1.24+ — Numerical arrays
- **SciPy** 1.10+ — Optimization solvers
- **CVXPY** 1.3+ — Convex optimization framework
- **OSQP** 0.6+ — Quadratic program solver
- **Pandas** — Data manipulation and CSV export

Install all:
```bash
pip install numpy scipy cvxpy osqp pandas
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

---

## Author

**Darién Mandujano** - ITAM (Instituto Tecnológico Autónomo de México)

Course: *Optimización Numérica I* (MAT-24431)

---

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

**Ready to run!** 🚀 Start with `python quick_demo.py`
