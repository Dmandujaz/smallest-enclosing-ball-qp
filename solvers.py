"""
QP Solvers for Smallest Enclosing Ball Problem

Implements multiple solvers:
1. CVXPY with different backends (OSQP, SCS, CLARABEL)
2. scipy.optimize (SLSQP, trust-constr)
3. Direct OSQP interface
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class SolverResult:
    """Store results from QP solver."""
    success: bool
    x: Optional[np.ndarray]  # Optimal solution
    objective_value: Optional[float]  # Optimal objective
    time: float  # Solve time in seconds
    iterations: Optional[int]  # Number of iterations
    solver_name: str
    status: str  # Solver status message
    
    
def solve_with_cvxpy(Q, q, A_eq, b_eq, bounds_lower, solver_name='OSQP', verbose=False):
    """
    Solve QP using CVXPY with specified backend.
    
    Problem:
        min  (1/2) x^T Q x + q^T x
        s.a. A_eq x = b_eq
             x >= bounds_lower
    
    Parameters:
    -----------
    Q : ndarray (n, n)
        Quadratic term matrix (must be PSD)
    q : ndarray (n,)
        Linear term vector
    A_eq : ndarray (m, n)
        Equality constraint matrix
    b_eq : ndarray (m,)
        Equality constraint RHS
    bounds_lower : ndarray (n,) or float
        Lower bounds on x (typically 0)
    solver_name : str
        CVXPY solver backend: 'OSQP', 'SCS', 'CLARABEL', 'ECOS'
    verbose : bool
        Print solver output
        
    Returns:
    --------
    result : SolverResult
    """
    import cvxpy as cp
    
    n = Q.shape[0]
    
    # Define variable
    x = cp.Variable(n)
    
    # Make Q symmetric (required by CVXPY)
    Q_sym = (Q + Q.T) / 2
    
    # Add small regularization to ensure PD (not just PSD) for quad_form
    # This is a common trick: Q_reg = Q + epsilon * I
    epsilon = 1e-8
    Q_reg = Q_sym + epsilon * np.eye(n)
    
    # Define objective
    # Using quadratic form: (1/2) x^T Q x + q^T x
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q_reg) + q @ x)
    
    # Define constraints
    constraints = [
        A_eq @ x == b_eq,
        x >= bounds_lower
    ]
    
    # Create problem
    problem = cp.Problem(objective, constraints)
    
    # Solve
    start_time = time.time()
    try:
        problem.solve(solver=solver_name, verbose=verbose)
        elapsed = time.time() - start_time
        
        # Extract results
        success = problem.status == cp.OPTIMAL
        x_opt = x.value if success else None
        obj_val = problem.value if success else None
        
        # Get iteration count if available
        iterations = None
        if hasattr(problem.solver_stats, 'num_iters'):
            iterations = problem.solver_stats.num_iters
        
        return SolverResult(
            success=success,
            x=x_opt,
            objective_value=obj_val,
            time=elapsed,
            iterations=iterations,
            solver_name=f"CVXPY-{solver_name}",
            status=problem.status
        )
        
    except Exception as e:
        elapsed = time.time() - start_time
        return SolverResult(
            success=False,
            x=None,
            objective_value=None,
            time=elapsed,
            iterations=None,
            solver_name=f"CVXPY-{solver_name}",
            status=f"Error: {str(e)}"
        )


def solve_with_scipy(Q, q, A_eq, b_eq, bounds_lower, method='SLSQP'):
    """
    Solve QP using scipy.optimize.
    
    Parameters:
    -----------
    Q, q, A_eq, b_eq, bounds_lower : as in solve_with_cvxpy
    method : str
        Scipy method: 'SLSQP' or 'trust-constr'
        
    Returns:
    --------
    result : SolverResult
    """
    from scipy.optimize import minimize, LinearConstraint, Bounds
    
    n = Q.shape[0]
    
    # Objective function and gradient
    def objective(x):
        return 0.5 * x @ Q @ x + q @ x
    
    def gradient(x):
        return Q @ x + q
    
    # Hessian (constant for QP)
    def hessian(x):
        return Q
    
    # Initial guess (feasible point)
    x0 = np.ones(n) / n  # Satisfies sum(x) = 1, x >= 0
    
    # Define constraints
    if method == 'SLSQP':
        constraints = {
            'type': 'eq',
            'fun': lambda x: A_eq @ x - b_eq,
            'jac': lambda x: A_eq
        }
        bounds = [(bounds_lower if np.isscalar(bounds_lower) else bounds_lower[i], None) 
                  for i in range(n)]
        
        start_time = time.time()
        try:
            res = minimize(
                objective,
                x0,
                method=method,
                jac=gradient,
                constraints=constraints,
                bounds=bounds,
                options={'ftol': 1e-9, 'disp': False}
            )
            elapsed = time.time() - start_time
            
            return SolverResult(
                success=res.success,
                x=res.x if res.success else None,
                objective_value=res.fun if res.success else None,
                time=elapsed,
                iterations=res.nit,
                solver_name=f"scipy-{method}",
                status=res.message
            )
        except Exception as e:
            elapsed = time.time() - start_time
            return SolverResult(
                success=False,
                x=None,
                objective_value=None,
                time=elapsed,
                iterations=None,
                solver_name=f"scipy-{method}",
                status=f"Error: {str(e)}"
            )
            
    elif method == 'trust-constr':
        # trust-constr uses different constraint format
        linear_constraint = LinearConstraint(A_eq, b_eq, b_eq)
        bounds_constraint = Bounds(
            lb=bounds_lower if np.isscalar(bounds_lower) else bounds_lower,
            ub=np.inf
        )
        
        start_time = time.time()
        try:
            res = minimize(
                objective,
                x0,
                method=method,
                jac=gradient,
                hess=hessian,
                constraints=linear_constraint,
                bounds=bounds_constraint,
                options={'verbose': 0}
            )
            elapsed = time.time() - start_time
            
            return SolverResult(
                success=res.success,
                x=res.x if res.success else None,
                objective_value=res.fun if res.success else None,
                time=elapsed,
                iterations=res.nit,
                solver_name=f"scipy-{method}",
                status=res.message
            )
        except Exception as e:
            elapsed = time.time() - start_time
            return SolverResult(
                success=False,
                x=None,
                objective_value=None,
                time=elapsed,
                iterations=None,
                solver_name=f"scipy-{method}",
                status=f"Error: {str(e)}"
            )
    else:
        raise ValueError(f"Unknown scipy method: {method}")


def solve_with_osqp_direct(Q, q, A_eq, b_eq, bounds_lower, verbose=False):
    """
    Solve QP using OSQP directly.
    
    OSQP solves:
        min  (1/2) x^T P x + q^T x
        s.a. l <= A x <= u
    
    We need to convert:
        A_eq x = b_eq  -->  b_eq <= A_eq x <= b_eq
        x >= bounds_lower  -->  bounds_lower <= I x <= inf
    
    Parameters:
    -----------
    Q, q, A_eq, b_eq, bounds_lower : as before
    verbose : bool
        
    Returns:
    --------
    result : SolverResult
    """
    import osqp
    from scipy import sparse
    
    n = Q.shape[0]
    
    # Convert Q to sparse (OSQP requires sparse matrices)
    P = sparse.csc_matrix(Q)
    
    # Stack constraints: [A_eq; I]
    I = sparse.eye(n)
    A = sparse.vstack([sparse.csc_matrix(A_eq), I], format='csc')
    
    # Bounds: [b_eq, b_eq] for equality, [bounds_lower, inf] for inequality
    l = np.concatenate([b_eq, np.full(n, bounds_lower if np.isscalar(bounds_lower) else bounds_lower)])
    u = np.concatenate([b_eq, np.full(n, np.inf)])
    
    # Create OSQP problem
    prob = osqp.OSQP()
    
    start_time = time.time()
    try:
        prob.setup(P, q, A, l, u, verbose=verbose, eps_abs=1e-6, eps_rel=1e-6)
        res = prob.solve()
        elapsed = time.time() - start_time
        
        success = res.info.status == 'solved'
        
        return SolverResult(
            success=success,
            x=res.x if success else None,
            objective_value=res.info.obj_val if success else None,
            time=elapsed,
            iterations=res.info.iter,
            solver_name="OSQP-direct",
            status=res.info.status
        )
    except Exception as e:
        elapsed = time.time() - start_time
        return SolverResult(
            success=False,
            x=None,
            objective_value=None,
            time=elapsed,
            iterations=None,
            solver_name="OSQP-direct",
            status=f"Error: {str(e)}"
        )


def solve_smallest_enclosing_ball(points, solver='CVXPY-OSQP', verbose=False):
    """
    High-level function to solve smallest enclosing ball using specified solver.
    
    Parameters:
    -----------
    points : ndarray, shape (d, n)
        Matrix of n points in R^d
    solver : str
        One of: 'CVXPY-OSQP', 'CVXPY-SCS', 'CVXPY-CLARABEL',
                'scipy-SLSQP', 'scipy-trustconstr', 'OSQP-direct'
    verbose : bool
        Print solver output
        
    Returns:
    --------
    result : SolverResult
    center : ndarray, shape (d,)
    radius : float
    active_points : ndarray
    """
    from generate_data import points_to_qp_format, extract_solution
    
    # Convert to QP format
    Q, q, A_eq, b_eq = points_to_qp_format(points)
    n = Q.shape[0]
    bounds_lower = 0.0  # x >= 0
    
    # Choose solver
    if solver.startswith('CVXPY-'):
        backend = solver.replace('CVXPY-', '')
        result = solve_with_cvxpy(Q, q, A_eq, b_eq, bounds_lower, backend, verbose)
    elif solver.startswith('scipy-'):
        method = solver.replace('scipy-', '')
        # Handle trust-constr special name
        if method == 'trustconstr':
            method = 'trust-constr'
        result = solve_with_scipy(Q, q, A_eq, b_eq, bounds_lower, method)
    elif solver == 'OSQP-direct':
        result = solve_with_osqp_direct(Q, q, A_eq, b_eq, bounds_lower, verbose)
    else:
        raise ValueError(f"Unknown solver: {solver}")
    
    # Extract geometric solution
    if result.success and result.x is not None:
        center, radius, active_points = extract_solution(
            result.x, points, result.objective_value
        )
    else:
        center, radius, active_points = None, None, None
    
    return result, center, radius, active_points


# List of available solvers
AVAILABLE_SOLVERS = [
    'CVXPY-OSQP',
    'CVXPY-SCS', 
    'scipy-SLSQP',
    'scipy-trustconstr',
    'OSQP-direct'
]


if __name__ == "__main__":
    print("Testing QP solvers...")
    from generate_data import generate_points
    
    # Generate small test problem
    n, d = 20, 3
    points = generate_points(n, d, distribution='uniform', seed=42)
    print(f"\nTest problem: {n} points in R^{d}")
    
    # Test each solver
    for solver in AVAILABLE_SOLVERS:
        print(f"\n{'='*60}")
        print(f"Testing {solver}...")
        print('='*60)
        
        try:
            result, center, radius, active = solve_smallest_enclosing_ball(
                points, solver=solver, verbose=False
            )
            
            print(f"Success: {result.success}")
            print(f"Time: {result.time:.4f} seconds")
            print(f"Iterations: {result.iterations}")
            print(f"Objective: {result.objective_value:.6f}" if result.objective_value else "N/A")
            print(f"Status: {result.status}")
            
            if result.success and center is not None:
                print(f"Center: {center[:3]}..." if d > 3 else f"Center: {center}")
                print(f"Radius: {radius:.6f}")
                print(f"Active points: {len(active)} (should be <= d+1 = {d+1})")
                
                # Verify solution
                distances = np.linalg.norm(points - center.reshape(-1, 1), axis=0)
                max_dist = np.max(distances)
                print(f"Max distance from center: {max_dist:.6f} (should equal radius)")
                print(f"Verification error: {abs(max_dist - radius):.2e}")
                
        except Exception as e:
            print(f"ERROR: {e}")
    
    print("\n" + "="*60)
    print("Solver testing complete!")
