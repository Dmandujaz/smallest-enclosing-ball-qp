"""
Main experimentation script for Smallest Enclosing Ball QP problem.

Runs experiments across dimensions d = 2, 3, ..., 30
with n = 50 points (fixed) and multiple random instances.
"""

import numpy as np
import pandas as pd
from generate_data import generate_points
from solvers import solve_smallest_enclosing_ball, AVAILABLE_SOLVERS
import time
from pathlib import Path


def run_single_experiment(n, d, solver, seed, distribution='uniform'):
    """
    Run a single experiment: generate data and solve.
    
    Returns dict with results.
    """
    # Generate points
    points = generate_points(n, d, distribution=distribution, seed=seed)
    
    # Solve
    result, center, radius, active = solve_smallest_enclosing_ball(
        points, solver=solver, verbose=False
    )
    
    # Verify solution
    max_error = None
    if result.success and center is not None:
        distances = np.linalg.norm(points - center.reshape(-1, 1), axis=0)
        max_dist = np.max(distances)
        max_error = abs(max_dist - radius)
    
    return {
        'dimension': d,
        'n_points': n,
        'trial': seed,
        'solver': result.solver_name,
        'success': result.success,
        'time': result.time,
        'iterations': result.iterations if result.iterations else -1,
        'objective': result.objective_value if result.objective_value else np.nan,
        'radius': radius if radius else np.nan,
        'n_active': len(active) if active is not None else -1,
        'max_error': max_error if max_error else np.nan,
        'status': result.status
    }


def run_experiments(dimensions, n_points=50, n_trials=20, 
                    solvers=None, distribution='uniform'):
    """
    Run complete experiment suite.
    
    Parameters:
    -----------
    dimensions : list or range
        Dimensions to test (e.g., range(2, 31))
    n_points : int
        Number of points (fixed)
    n_trials : int
        Number of random instances per dimension
    solvers : list of str
        Solvers to use (default: all available)
    distribution : str
        Point distribution type
        
    Returns:
    --------
    results_df : DataFrame
        Complete results
    """
    if solvers is None:
        solvers = AVAILABLE_SOLVERS
    
    results = []
    total = len(dimensions) * n_trials * len(solvers)
    current = 0
    
    print(f"{'='*70}")
    print(f"EXPERIMENT CONFIGURATION")
    print(f"{'='*70}")
    print(f"Dimensions: {list(dimensions)}")
    print(f"Points per instance: {n_points}")
    print(f"Trials per dimension: {n_trials}")
    print(f"Solvers: {solvers}")
    print(f"Distribution: {distribution}")
    print(f"Total experiments: {total}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    for d in dimensions:
        print(f"\n{'='*70}")
        print(f"DIMENSION d = {d}")
        print(f"{'='*70}")
        
        for trial in range(n_trials):
            seed = trial  # Use trial number as seed for reproducibility
            
            for solver in solvers:
                current += 1
                
                # Progress indicator
                if current % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / current
                    eta = avg_time * (total - current)
                    print(f"Progress: {current}/{total} ({100*current/total:.1f}%) | "
                          f"ETA: {eta/60:.1f} min")
                
                try:
                    result = run_single_experiment(
                        n_points, d, solver, seed, distribution
                    )
                    results.append(result)
                    
                except Exception as e:
                    print(f"ERROR in d={d}, trial={trial}, solver={solver}: {e}")
                    results.append({
                        'dimension': d,
                        'n_points': n_points,
                        'trial': trial,
                        'solver': solver,
                        'success': False,
                        'time': np.nan,
                        'iterations': -1,
                        'objective': np.nan,
                        'radius': np.nan,
                        'n_active': -1,
                        'max_error': np.nan,
                        'status': f'Error: {e}'
                    })
        
        # Print dimension summary
        df_d = pd.DataFrame(results)
        df_d = df_d[df_d['dimension'] == d]
        print(f"\nDimension {d} summary:")
        for solver in solvers:
            df_s = df_d[df_d['solver'] == solver]
            if len(df_s) > 0:
                success_rate = df_s['success'].mean()
                mean_time = df_s[df_s['success']]['time'].mean()
                mean_iters = df_s[df_s['success']]['iterations'].mean()
                print(f"  {solver:20s}: success={success_rate:.1%}, "
                      f"time={mean_time:.4f}s, iters={mean_iters:.0f}")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Total time: {elapsed/60:.2f} minutes")
    print(f"{'='*70}")
    
    return pd.DataFrame(results)


def compute_summary_statistics(results_df):
    """
    Compute summary statistics from results.
    
    Returns DataFrame with statistics grouped by dimension and solver.
    """
    # Filter only successful runs
    df_success = results_df[results_df['success'] == True].copy()
    
    # Group by dimension and solver
    grouped = df_success.groupby(['dimension', 'solver'])
    
    summary = grouped.agg({
        'time': ['mean', 'std', 'median', 'min', 'max'],
        'iterations': ['mean', 'std', 'median'],
        'n_active': ['mean', 'std'],
        'max_error': ['mean', 'max'],
        'success': 'count'  # Number of successful runs
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary.rename(columns={'success_count': 'n_runs'}, inplace=True)
    
    return summary


def save_results(results_df, summary_df, output_dir='/mnt/user-data/outputs'):
    """Save results to CSV files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    results_path = Path(output_dir) / 'qp_results_raw.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nRaw results saved to: {results_path}")
    
    # Save summary statistics
    summary_path = Path(output_dir) / 'qp_results_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")
    
    return results_path, summary_path


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" SMALLEST ENCLOSING BALL - QP EXPERIMENTATION")
    print("="*70 + "\n")
    
    # Configuration
    DIMENSIONS = range(2, 31)  # d = 2, 3, ..., 30
    N_POINTS = 50
    N_TRIALS = 20
    DISTRIBUTION = 'uniform'
    
    # Select solvers (use fastest ones for full experiment)
    SOLVERS_TO_USE = [
        'CVXPY-OSQP',
        'scipy-SLSQP',
        'OSQP-direct'
    ]
    
    print("Starting experiments...")
    print("This will take several minutes...\n")
    
    # Run experiments
    results_df = run_experiments(
        dimensions=DIMENSIONS,
        n_points=N_POINTS,
        n_trials=N_TRIALS,
        solvers=SOLVERS_TO_USE,
        distribution=DISTRIBUTION
    )
    
    # Compute statistics
    print("\nComputing summary statistics...")
    summary_df = compute_summary_statistics(results_df)
    
    # Display sample statistics
    print("\nSample statistics (first few dimensions):")
    print(summary_df.head(15).to_string(index=False))
    
    # Save results
    save_results(results_df, summary_df)
    
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
