"""
Quick Demo: Run a small-scale experiment for testing/demonstration.

This is useful for:
- Testing the setup
- Getting quick results for presentation
- Debugging before full experiment
"""

from run_experiments import run_experiments, compute_summary_statistics, save_results
import pandas as pd

print("\n" + "="*70)
print(" QUICK DEMO: Smallest Enclosing Ball QP")
print("="*70 + "\n")

# Run small experiment
print("Running small-scale experiment...")
print("This should take < 1 minute\n")

results = run_experiments(
    dimensions=range(2, 11, 2),  # d = 2, 4, 6, 8, 10
    n_points=50,
    n_trials=5,  # Only 5 trials instead of 20
    solvers=['CVXPY-OSQP', 'scipy-SLSQP'],  # Two fastest solvers
    distribution='uniform'
)

# Compute statistics
summary = compute_summary_statistics(results)

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(summary.to_string(index=False))

# Save results
save_results(results, summary)

print("\n" + "="*70)
print("Key Observations:")
print("="*70)

# Compute some interesting metrics
for solver in ['CVXPY-OSQP', 'scipy-SLSQP']:
    solver_data = results[results['solver'] == solver]
    if len(solver_data) > 0:
        avg_time = solver_data['time'].mean()
        print(f"\n{solver}:")
        print(f"  Average time: {avg_time*1000:.2f} ms")
        print(f"  Success rate: {solver_data['success'].mean()*100:.0f}%")
        
        # Check theoretical property: n_active <= d+1
        valid = solver_data[solver_data['success'] == True]
        if len(valid) > 0:
            max_active = valid.groupby('dimension')['n_active'].mean()
            theory_limit = valid.groupby('dimension')['dimension'].first() + 1
            violation = (max_active > theory_limit).sum()
            print(f"  Avg active points vs d+1 limit: {max_active.mean():.1f} vs {theory_limit.mean():.1f}")
            print(f"  Violations of d+1 limit: {violation}/{len(max_active)}")

print("\n" + "="*70)
print("QUICK DEMO COMPLETE!")
print("="*70)
print("\nFor full experiment, run: python run_experiments.py")
print("For custom experiments, see README.md\n")
